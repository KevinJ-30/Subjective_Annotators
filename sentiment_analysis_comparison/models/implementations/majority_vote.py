import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from .base_model import BaseModel

class MajorityVoteModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.num_classes

        # Store majority votes per sample index
        self.majority_votes = {}
        # Dummy parameter so we can return a float loss tensor
        self.dummy_parameter = nn.Parameter(torch.zeros(1))

    def compute_majority_vote(self, sample_indices, labels):
        """
        sample_indices: Tensor[B] of sample IDs
        labels:         Tensor[B] of ints in [0, num_classes)
        """
        buckets = defaultdict(list)
        for idx, lab in zip(sample_indices, labels):
            buckets[int(idx.item())].append(int(lab.item()))
        majority = {}
        for idx, labs in buckets.items():
            counts = np.bincount(labs, minlength=self.num_classes)
            majority[idx] = int(np.argmax(counts))
        return majority

    def forward(self, input_ids, attention_mask, annotator_id,
                label=None, sample_index=None):
        # ── TRAINING ─────────────────────────────────────────────────────
        if self.training and label is not None:
            # update stored votes if we have sample indices
            if sample_index is not None:
                votes = self.compute_majority_vote(sample_index, label)
                self.majority_votes.update(votes)
            # return a zero float loss so DeepSpeed can backward()
            return self.dummy_parameter.sum() * 0.0

        # ── INFERENCE ────────────────────────────────────────────────────
        # figure out which sample IDs to look up
        if sample_index is not None:
            indices = sample_index
        else:
            # fallback: use first token ID just to keep shape
            indices = input_ids[:, 0]

        # build a Python list of votes
        preds_list = []
        for idx in indices.cpu().tolist():
            preds_list.append(self.majority_votes.get(int(idx), 0))

        # tensor-ify, and ensure at least 1-D
        out = torch.tensor(preds_list, dtype=torch.long, device=self.device)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out
