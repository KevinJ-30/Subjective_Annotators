"""
Precompute RoBERTa [CLS] embeddings for the HSB-binary dataset.

Run once before noise experiments:
    python scripts/compute_embeddings.py \
        --data_path data/hsb_brexit/processed/train.json \
        --output_path data/hsb_brexit/processed/train_embeddings.npy

Produces two files:
    <output_path>          — float32 array (N_unique_instances, 768)
    <output_path stem>_ids.npy — object array (N_unique_instances,) of uid values
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel


TEXT_COL = 'question'
ID_COL = 'uid'
MODEL_NAME = 'roberta-base'
BATCH_SIZE = 64
MAX_LENGTH = 128


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def compute_embeddings(data_path: str, output_path: str, batch_size: int = BATCH_SIZE,
                       max_length: int = MAX_LENGTH, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    df = pd.read_json(data_path, lines=True)
    unique_df = df.drop_duplicates(subset=[ID_COL])[[ID_COL, TEXT_COL]].reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {len(unique_df)} unique instances")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    dataset = TextDataset(unique_df[TEXT_COL].tolist(), tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_embeddings.append(cls_embeddings)
            print(f"  Processed {sum(len(e) for e in all_embeddings)}/{len(unique_df)} instances", end='\r')

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"\nEmbeddings shape: {embeddings.shape}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)

    ids_path = out.parent / (out.stem + '_ids.npy')
    np.save(ids_path, unique_df[ID_COL].values)

    print(f"Saved embeddings to {out}")
    print(f"Saved instance IDs to {ids_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute RoBERTa [CLS] embeddings for HSB-binary')
    parser.add_argument('--data_path', type=str,
                        default='data/hsb_brexit/processed/train.json',
                        help='Path to train.json (jsonlines)')
    parser.add_argument('--output_path', type=str,
                        default='data/hsb_brexit/processed/train_embeddings.npy',
                        help='Output path for embeddings array')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH)
    parser.add_argument('--device', type=str, default=None,
                        help='cuda or cpu (auto-detected if omitted)')
    args = parser.parse_args()

    compute_embeddings(
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )


if __name__ == '__main__':
    main()
