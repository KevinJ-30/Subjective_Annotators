from pathlib import Path
from datasets import load_dataset
import pandas as pd
import torch  # kept because your project imports it elsewhere sometimes
import numpy as np


def _split_by_item(
    df: pd.DataFrame, item_col: str, test_ratio: float = 0.2, seed: int = 42
):
    """
    Deterministic train/test split by item id to prevent leakage across annotators.
    All annotations for an item go to the same split.
    """
    rng = np.random.RandomState(seed)
    items = df[item_col].unique()
    rng.shuffle(items)

    n_items = len(items)
    n_test = max(1, int(test_ratio * n_items))

    test_items = set(items[:n_test])
    train_items = set(items[n_test:])

    train_df = df[df[item_col].isin(train_items)].reset_index(drop=True)
    test_df = df[df[item_col].isin(test_items)].reset_index(drop=True)
    return train_df, test_df


def download_epic_dataset():
    """
    Downloads and preprocesses the EPIC dataset.

    BEHAVIOR:
      - Downloads EPIC from Hugging Face
      - Adapts EPIC into the exact schema your current pipeline expects
      - Saves it to: data/epic_dataset/processed/{train,test}.json

    Output columns (minimum required by data_loader.py):
      - question (str): parent_text </s> reply_text
      - answer_label (int): label id (0/1)
      - annotator_id (str): annotator identity
      - original_id (int): per-item numeric id that is safe to cast to int
    Additional columns retained (safe for debugging):
      - answer (str): "not"/"iro"
      - original_id_raw (str): EPIC id_original as string
      - domain (str): EPIC source (e.g., reddit)
      - task (str): "offensiveness detection" (kept to match md setting if other code uses it)
    """
    print("Downloading EPIC dataset...")

    # Keep the same output directory structure used by the rest of your code
    base_path = Path("data/epic_dataset")
    processed_path = base_path / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    # Load EPIC
    dataset = load_dataset("Multilingual-Perspectivist-NLU/EPIC")
    ds = dataset["train"]  # EPIC provides a single split

    df = pd.DataFrame(ds)

    # Sanity-check required EPIC fields
    required_epic_cols = ["user", "label", "id_original", "text", "parent_text"]
    missing = [c for c in required_epic_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"EPIC missing expected columns: {missing}. Found: {list(df.columns)}"
        )

    # ---- Create the MD-shaped fields expected by your existing loader ----

    # question = context + reply (AART-style)
    df["question"] = df["parent_text"].fillna("") + "</s>" + df["text"].fillna("")

    # annotator_id
    df["annotator_id"] = df["user"].astype(str)

    # label mapping: AART-compatible convention
    # "not" -> 0, "iro" -> 1
    # EPIC viewer shows these exact strings; we handle both string and ClassLabel cases.
    label_feature = ds.features.get("label", None)
    if label_feature is not None and hasattr(label_feature, "names"):
        # ClassLabel: stored as ints already; keep them
        labels_int = ds["label"]
        labels_str = [label_feature.int2str(i) for i in labels_int]
        df["answer_label"] = labels_int
        df["answer"] = labels_str
    else:
        mapping = {"iro": 0, "not": 1}
        if not set(df["label"].unique()).issubset(set(mapping.keys())):
            # fallback deterministic mapping
            unique_labels = sorted(df["label"].unique())
            mapping = {lab: i for i, lab in enumerate(unique_labels)}
        df["answer_label"] = df["label"].map(mapping).astype(int)
        df["answer"] = df["label"]

    # original_id: must be numeric because your loader casts it to int
    # EPIC id_original can be huge (and may overflow int64). So:
    # - keep raw as string
    # - create safe contiguous int id per unique item
    df["original_id_raw"] = df["id_original"].astype(str)
    unique_items = sorted(df["original_id_raw"].unique())
    item2idx = {item: idx for idx, item in enumerate(unique_items)}
    df["original_id"] = df["original_id_raw"].map(item2idx).astype(int)

    # Optional fields that other parts of your pipeline might rely on
    # Keep task name EXACTLY as md-agreement dataset uses, to avoid surprises in downstream logging/filters.
    df["task"] = "offensiveness detection"
    df["domain"] = df["source"].fillna("") if "source" in df.columns else ""

    # Keep the same split names as before
    keep_cols = [
        "task",
        "domain",
        "question",
        "original_id",
        "annotator_id",
        "answer_label",
        "answer",
        "original_id_raw",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    # Split by original_id (item-level split)
    train_df, test_df = _split_by_item(
        df, item_col="original_id", test_ratio=0.2, seed=42
    )

    # Save as the same files your project already expects
    splits = {"train": train_df, "test": test_df}
    for split_name, split_df in splits.items():
        output_file = processed_path / f"{split_name}.json"

        # Overwrite to ensure you're actually switching datasets (intentional)
        print(
            f"\nWriting {split_name} split to {output_file} (overwriting if exists)..."
        )

        print(f"Examples: {len(split_df)}")
        print(f"Unique annotators: {split_df['annotator_id'].nunique()}")
        print(f"Unique items: {split_df['original_id'].nunique()}")
        print(
            f"Label dist: {split_df['answer_label'].value_counts(normalize=True).to_dict()}"
        )

        split_df.to_json(output_file, orient="records", lines=True)

    print(
        "\nEPIC download + preprocessing completed; data written to data/epic_dataset/processed/"
    )


if __name__ == "__main__":
    download_epic_dataset()
