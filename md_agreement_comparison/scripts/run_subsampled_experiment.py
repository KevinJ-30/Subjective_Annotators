import os
import sys
import torch
import json
from pathlib import Path
from datetime import datetime
import argparse
import logging
import pandas as pd
import numpy as np
import uuid
import traceback

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/subsampled_experiment.log'),
        logging.StreamHandler()
    ]
)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.config import ExperimentConfig
from scripts.train import Trainer
from models.implementations.multitask import MultitaskModel
from models.implementations.aart import AARTModel
from models.implementations.annotator_embedding import AnnotatorEmbeddingModel

def get_experiment_id(args):
    """Generate a unique experiment ID based on parameters"""
    # Create a descriptive name
    name_parts = []
    
    # Add approaches
    name_parts.append(f"approach-{args.approach}")
    
    # Add annotator subsampling info
    name_parts.append(f"n_annotators-{args.n_annotators}")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts.append(timestamp)
    
    # Add unique ID
    unique_id = str(uuid.uuid4())[:8]
    name_parts.append(unique_id)
    
    # Join all parts with double underscore
    return "__".join(name_parts)

def subsample_annotators(data_path, n_annotators, random_state=42):
    """
    Efficiently subsample n_annotators from the dataset by reading in chunks.
    
    Args:
        data_path (str): Path to the JSONL file
        n_annotators (int): Number of annotators to keep
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Dataset with only the selected annotators
        list: List of selected annotator IDs
    """
    # First pass: get unique annotators
    unique_annotators = set()
    for chunk in pd.read_json(data_path, lines=True, chunksize=10000):
        unique_annotators.update(chunk['annotator_id'].unique())
    unique_annotators = list(unique_annotators)
    
    if n_annotators >= len(unique_annotators):
        print(f"Requested {n_annotators} annotators but only {len(unique_annotators)} available. Using all annotators.")
        return data_path, unique_annotators
    
    # Randomly select n_annotators
    np.random.seed(random_state)
    selected_annotators = np.random.choice(unique_annotators, n_annotators, replace=False)
    selected_annotators = set(selected_annotators)
    
    # Second pass: filter data
    filtered_data = []
    for chunk in pd.read_json(data_path, lines=True, chunksize=10000):
        chunk_filtered = chunk[chunk['annotator_id'].isin(selected_annotators)]
        if not chunk_filtered.empty:
            filtered_data.append(chunk_filtered)
    
    if not filtered_data:
        raise ValueError("No data found for selected annotators")
    
    filtered_df = pd.concat(filtered_data, ignore_index=True)
    
    # Log statistics
    print(f"\nAnnotator Subsampling Statistics:")
    print(f"Original number of annotators: {len(unique_annotators)}")
    print(f"Selected number of annotators: {len(selected_annotators)}")
    print(f"Selected annotators: {sorted(selected_annotators)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    
    return filtered_df, list(selected_annotators)

def run_experiment(approach, n_annotators, experiment_id):
    """Run experiment with subsampled annotators"""
    try:
        logging.info(f"\nStarting experiment for {approach} with {n_annotators} annotators")
        
        # Create config
        config = ExperimentConfig(approach=approach)
        config.model_name = 'bert-base-uncased'
        config.max_length = 128
        config.batch_size = 16  # Reduced batch size to help with memory
        config.learning_rate = 2e-5
        config.num_epochs = 10
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.n_gpu = torch.cuda.device_count()
        
        # Set experiment ID and directories
        config.experiment_id = experiment_id
        config.checkpoint_dir = Path(f"experiments/{experiment_id}/models/checkpoints/{approach}")
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Subsample data
        train_data, selected_annotators = subsample_annotators('data/md_agreement/processed/train.json', n_annotators)
        test_data, _ = subsample_annotators('data/md_agreement/processed/test.json', n_annotators)
        
        # Save subsampled data
        data_dir = Path(f"experiments/{experiment_id}/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        train_data.to_json(data_dir / "train.json", orient='records', lines=True)
        test_data.to_json(data_dir / "test.json", orient='records', lines=True)
        
        # Update config with data paths
        config.train_path = str(data_dir / "train.json")
        config.test_path = str(data_dir / "test.json")
        
        # Create trainer with appropriate model
        if approach == 'aart':
            trainer = Trainer(config, AARTModel)
        elif approach == 'multitask':
            trainer = Trainer(config, MultitaskModel)
        elif approach == 'annotator_embedding':
            trainer = Trainer(config, AnnotatorEmbeddingModel)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        # Train model
        results = trainer.train()
        
        # Save results
        results_dir = Path(f"experiments/{experiment_id}/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary with annotator information
        summary = {
            "approach": approach,
            "n_annotators": n_annotators,
            "selected_annotators": selected_annotators,
            "metrics": results
        }
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return results
        
    except Exception as e:
        logging.error(f"Error running experiment for {approach}: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description='Run experiment with subsampled annotators')
    parser.add_argument('--approach', type=str, required=True,
                      choices=['aart', 'multitask', 'annotator_embedding'],
                      help='Approach to evaluate')
    parser.add_argument('--n_annotators', type=int, required=True,
                      help='Number of annotators to use for training')
    args = parser.parse_args()
    
    # Generate experiment ID
    experiment_id = get_experiment_id(args)
    logging.info(f"Starting experiment with ID: {experiment_id}")
    
    # Run experiment
    results = run_experiment(args.approach, args.n_annotators, experiment_id)
    
    # Print final summary
    print("\n=== Final Results Summary ===")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n=== {args.approach.upper()} ===")
    if results is not None:
        print(json.dumps(results, indent=2))
    else:
        print("No results available")
    
    print("\n=== End of Summary ===")

if __name__ == "__main__":
    main() 