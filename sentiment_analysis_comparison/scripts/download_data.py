from pathlib import Path
from datasets import load_dataset
import pandas as pd
import torch
import numpy as np

def download_sentiment_data():
    """
    Download Sentiment Analysis dataset from Hugging Face datasets
    """
    print("Downloading Sentiment Analysis dataset...")
    
    # Create data directories
    base_path = Path("data/sentiment_analysis")
    processed_path = base_path / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    # Define label mapping
    label_map = {
        'Very negative': 0,
        'Somewhat negative': 1,
        'Neutral': 2,
        'Somewhat positive': 3,
        'Very positive': 4
    }

    try:
        # Load dataset with the correct configuration
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("MichiganNLP/TID-8", "sentiment-ann")
        print("Available splits:", dataset.keys())
        
        # Print dataset info
        print("\nDataset Info:")
        print(dataset)
        print("\nFeatures:")
        print(dataset['train'].features)
        
        # Convert and save each split
        splits = {
            'train': dataset['train'],
            'test': dataset['test']
        }
        
        for split_name, split_data in splits.items():
            output_file = processed_path / f"{split_name}.json"
            
            if not output_file.exists():
                print(f"\nProcessing {split_name} split...")
                
                # Convert to pandas DataFrame
                df = pd.DataFrame(split_data)
                
                # Map text labels to numeric values
                df['answer_label'] = df['answer'].map(label_map)
                
                # Print dataset statistics
                print(f"\nStatistics for {split_name} split:")
                print(f"Number of examples: {len(df)}")
                print(f"Number of unique annotators: {len(df['annotator_id'].unique())}")
                print(f"Label distribution: {df['answer_label'].value_counts(normalize=True).to_dict()}")
                
                # Save to JSON
                df.to_json(output_file, orient='records', lines=True)
                print(f"Saved {split_name} split to {output_file}")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

    print("\nDataset download and processing completed!")

if __name__ == "__main__":
    download_sentiment_data() 