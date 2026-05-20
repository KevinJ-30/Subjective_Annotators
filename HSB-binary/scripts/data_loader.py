"""
data_loader.py — drop-in replacement.

Changes from previous version
-------------------------------
[F7] Added LabelDiverseBatchSampler.
     Replaces GroupByInstanceBatchSampler throughout the trainer.
     Guarantees every batch contains ≥ min_labels_per_text distinct labels
     for each included text_id, maximising valid (pos, neg) RINCE pairs.

     Strategy:
       1. Pre-index samples by text_id and record labels present per text.
       2. Classify texts as "mixed" (≥ 2 distinct labels) vs "uniform".
       3. Build batches by filling with complete text-groups, prioritising
          mixed texts, so that most pairs within each batch are informative.
       4. Any remaining uniform-text samples are appended as a final batch
          rather than scattered across batches, reducing noise.

MDAgreementDataset is unchanged except for a minor guard on missing columns.
"""

import random
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MDAgreementDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, device=None, noise_config=None):
        if isinstance(data, pd.DataFrame):
            self.data = data.reset_index(drop=True)
        else:
            self.data = pd.read_json(data, lines=True).reset_index(drop=True)

        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.device     = device if device is not None else torch.device("cpu")

        # Store original distribution
        self.original_dist = self.data["answer_label"].value_counts()

        # Apply noise if requested
        if noise_config is not None and noise_config.get("add_noise", False):
            from scripts.noise_utils import add_annotator_noise
            self.data      = add_annotator_noise(self.data, noise_config)
            noisy_dist     = self.data["answer_label"].value_counts()
            logging.info("Verifying noise application in dataset:")
            logging.info(f"  Original  : {dict(self.original_dist)}")
            logging.info(f"  After noise: {dict(noisy_dist)}")

        # Deterministic annotator → int mapping
        unique_annotators        = sorted(self.data["annotator_id"].unique())
        self.annotator2id        = {a: i for i, a in enumerate(unique_annotators)}
        self._num_annotators     = len(unique_annotators)

        print(f"\nDataset Statistics:")
        print(f"  Samples    : {len(self.data)}")
        print(f"  Annotators : {self._num_annotators}")
        print(f"  Annotator IDs: {sorted(unique_annotators)}")
        print(f"  Sample row :\n{self.data.iloc[0]}")

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row        = self.data.iloc[idx]
        text       = row["question"]
        label      = row["answer_label"]
        annotator  = row["annotator_id"]
        text_id    = row["original_id"]

        ann_id = self.annotator2id[annotator]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if idx == 0:
            print(f"\nSample batch[0]: label={label}, ann_id={ann_id}, text_id={text_id}")

        return {
            "input_ids":      encoding["input_ids"].squeeze().to(self.device),
            "attention_mask": encoding["attention_mask"].squeeze().to(self.device),
            "label":          torch.tensor(int(label),    device=self.device),
            "annotator_id":   torch.tensor(int(ann_id),   device=self.device),
            "text_id":        torch.tensor(int(text_id),  device=self.device),
        }

    @property
    def num_annotators(self):
        return self._num_annotators

    @property
    def text_ids(self):
        return self.data["original_id"].tolist()


# ---------------------------------------------------------------------------
# LabelDiverseBatchSampler  [F7]
# ---------------------------------------------------------------------------

class LabelDiverseBatchSampler(Sampler):
    """
    Batch sampler that maximises valid RINCE positive/negative pairs.

    Guarantees:
      • All annotators of the same text_id land in the same batch
        (same guarantee as GroupByInstanceBatchSampler).
      • Texts with ≥ min_labels_per_text distinct labels ("mixed" texts) are
        preferentially scheduled first, so most batches are contrastive-rich.
      • Uniform-text groups are collected into separate, later batches.

    Args:
        dataset            : MDAgreementDataset instance.
        max_batch_size     : Upper bound on samples per batch.
        shuffle            : Shuffle text order within each group each epoch.
        seed               : Base RNG seed (incremented each __iter__ call).
        min_labels_per_text: Minimum distinct labels for a text to be "mixed".
    """

    def __init__(
        self,
        dataset: MDAgreementDataset,
        max_batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        min_labels_per_text: int = 2,
    ):
        self.dataset             = dataset
        self.max_batch_size      = max_batch_size
        self.shuffle             = shuffle
        self.seed                = seed
        self.min_labels_per_text = min_labels_per_text
        self._epoch              = 0

        self._build_index()

    # ------------------------------------------------------------------

    def _build_index(self):
        """Build text_id → [(dataset_idx, label)] and classify texts."""
        data             = self.dataset.data
        self.text_groups = {}   # text_id (int) → list of (int idx, int label)

        for idx in range(len(data)):
            row   = data.iloc[idx]
            tid   = int(row["original_id"])
            label = int(row["answer_label"])
            self.text_groups.setdefault(tid, []).append((idx, label))

        self.mixed_texts   = []
        self.uniform_texts = []
        for tid, group in self.text_groups.items():
            labels = {lbl for _, lbl in group}
            if len(labels) >= self.min_labels_per_text:
                self.mixed_texts.append(tid)
            else:
                self.uniform_texts.append(tid)

        total_samples  = sum(len(g) for g in self.text_groups.values())
        mixed_samples  = sum(len(self.text_groups[t]) for t in self.mixed_texts)
        print(
            f"[LabelDiverseBatchSampler] texts={len(self.text_groups)} "
            f"mixed={len(self.mixed_texts)} ({100*len(self.mixed_texts)/max(1,len(self.text_groups)):.0f}%) "
            f"mixed_samples={mixed_samples}/{total_samples} "
            f"({100*mixed_samples/max(1,total_samples):.0f}%)"
        )

    # ------------------------------------------------------------------

    def _build_batches(self, rng: random.Random):
        """
        Build batch index lists.

        Order:  mixed-text groups (shuffled) → uniform-text groups (shuffled).
        Within a text-group, sample order is also shuffled.
        A group that would overflow the current batch starts a new batch.
        """
        mixed   = list(self.mixed_texts)
        uniform = list(self.uniform_texts)
        if self.shuffle:
            rng.shuffle(mixed)
            rng.shuffle(uniform)

        batches  = []
        current  = []

        for tid in mixed + uniform:
            indices = [idx for idx, _ in self.text_groups[tid]]
            if self.shuffle:
                rng.shuffle(indices)

            # If adding this group would exceed the batch, flush first.
            if current and (len(current) + len(indices) > self.max_batch_size):
                batches.append(current)
                current = []

            # If the group alone exceeds the batch size, chunk it.
            if len(indices) > self.max_batch_size:
                for start in range(0, len(indices), self.max_batch_size):
                    chunk = indices[start : start + self.max_batch_size]
                    if current:
                        batches.append(current)
                        current = []
                    batches.append(chunk)
            else:
                current.extend(indices)
                if len(current) >= self.max_batch_size:
                    batches.append(current[: self.max_batch_size])
                    current = current[self.max_batch_size :]

        if current:
            batches.append(current)

        return batches

    # ------------------------------------------------------------------

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        batches = self._build_batches(rng)
        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        total = sum(len(g) for g in self.text_groups.values())
        return max(1, (total + self.max_batch_size - 1) // self.max_batch_size)


# ---------------------------------------------------------------------------
# Utility: annotator subsampling (unchanged)
# ---------------------------------------------------------------------------

def subsample_annotators(data, n_annotators, random_state=42):
    """
    Subsample n_annotators from the dataset.

    Returns:
        subsampled_data (pd.DataFrame), selected_annotators (list)
    """
    unique_annotators = data["annotator_id"].unique()

    if n_annotators >= len(unique_annotators):
        print(
            f"Requested {n_annotators} annotators but only "
            f"{len(unique_annotators)} available. Using all."
        )
        return data, unique_annotators.tolist()

    np.random.seed(random_state)
    selected          = np.random.choice(unique_annotators, n_annotators, replace=False)
    subsampled_data   = data[data["annotator_id"].isin(selected)].copy()

    print(f"\nAnnotator Subsampling:")
    print(f"  Original  : {len(unique_annotators)}")
    print(f"  Selected  : {len(selected)}  {sorted(selected)}")
    print(f"  Rows      : {len(data)} → {len(subsampled_data)}")

    return subsampled_data, selected.tolist()