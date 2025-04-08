from pathlib import Path
from datasets import load_dataset
import pandas as pd
import torch

def download_md_agreement():
    """
    Download MD Agreement dataset from Hugging Face datasets
    """
    print("Downloading MD Agreement dataset...")
    
    # Create data directories
    base_path = Path("data/md_agreement")
    processed_path = base_path / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    # Load dataset with the correct configuration
    dataset = load_dataset("MichiganNLP/TID-8", "md-agreement-ann")
    print("Available splits:", dataset.keys())
    
    # Convert and save each split
    splits = {
        'train': dataset['train'],
        'test': dataset['test']
    }
    
    for split_name, split_data in splits.items():
        output_file = processed_path / f"{split_name}.json"
        
        if not output_file.exists():
            print(f"Processing {split_name} split...")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(split_data)
            
            # Print dataset statistics
            print(f"\nStatistics for {split_name} split:")
            print(f"Number of examples: {len(df)}")
            print(f"Number of unique annotators: {len(df['annotator_id'].unique())}")
            print(f"Label distribution: {df['label'].value_counts(normalize=True).to_dict()}")
            
            # Save to JSON
            df.to_json(output_file, orient='records', lines=True)
            print(f"Saved {split_name} split to {output_file}")

    print("\nDataset download and processing completed!")

if __name__ == "__main__":
    download_md_agreement()