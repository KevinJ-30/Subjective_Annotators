import os
import sys
import torch
import json
from pathlib import Path
from datetime import datetime
import argparse
import logging
import numpy as np
import pandas as pd
import uuid
import traceback
from sklearn.model_selection import train_test_split

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/annotator_subsampling.log'),
        logging.StreamHandler()
    ]
)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.config import ExperimentConfig
from scripts.subsampling_trainer import SubsamplingTrainer
from scripts.metrics import evaluate_model
from scripts.data_loader import MDAgreementDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Import models
from models.implementations.multitask import MultitaskModel
from models.implementations.aart import AARTModel
from models.implementations.annotator_embedding import AnnotatorEmbeddingModel

def get_experiment_id(args):
    """Generate a unique experiment ID based on parameters"""
    # Create a descriptive name
    name_parts = []
    
    # Add approaches
    approach_str = '_'.join(sorted(args.approaches))
    name_parts.append(f"approaches-{approach_str}")
    
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

def subsample_annotators(data, n_annotators, random_state=42):
    """
    Subsample n_annotators from the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        n_annotators (int): Number of annotators to keep
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Dataset with only the selected annotators
        list: List of selected annotator IDs
    """
    # Get unique annotators
    unique_annotators = data['annotator_id'].unique()
    
    if n_annotators >= len(unique_annotators):
        logging.info(f"Requested {n_annotators} annotators but only {len(unique_annotators)} available. Using all annotators.")
        return data, unique_annotators.tolist()
    
    # Randomly select n_annotators
    np.random.seed(random_state)
    selected_annotators = np.random.choice(unique_annotators, n_annotators, replace=False)
    
    # Filter data to only include selected annotators
    subsampled_data = data[data['annotator_id'].isin(selected_annotators)].copy()
    
    # Log statistics
    logging.info(f"\nAnnotator Subsampling Statistics:")
    logging.info(f"Original number of annotators: {len(unique_annotators)}")
    logging.info(f"Selected number of annotators: {len(selected_annotators)}")
    logging.info(f"Selected annotators: {sorted(selected_annotators)}")
    logging.info(f"Original dataset size: {len(data)}")
    logging.info(f"Subsampled dataset size: {len(subsampled_data)}")
    
    return subsampled_data, selected_annotators.tolist()

def run_single_experiment(approach, experiment_id, n_annotators):
    """Run a single experiment with the specified approach and number of annotators"""
    try:
        logging.info(f"\nStarting experiment for {approach} with {n_annotators} annotators")
        
        # Create config with approach
        config = ExperimentConfig(approach=approach)
        config.model_name = 'bert-base-uncased'
        config.max_length = 128
        config.batch_size = 32
        config.learning_rate = 2e-5
        config.num_epochs = 10
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.n_gpu = torch.cuda.device_count()
        
        # Set experiment ID and directories
        config.experiment_id = experiment_id
        config.checkpoint_dir = Path(f"experiments/{experiment_id}/models/checkpoints/{approach}")
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        train_path = 'data/md_agreement/processed/train.json'
        test_path = 'data/md_agreement/processed/test.json'
        
        # Load data into DataFrames
        train_data = pd.read_json(train_path, lines=True)
        test_data = pd.read_json(test_path, lines=True)
        
        # Subsample annotators for both training and test data
        # Use the same random state to ensure same annotators are selected
        train_data, selected_annotators = subsample_annotators(train_data, n_annotators)
        test_data = test_data[test_data['annotator_id'].isin(selected_annotators)].copy()
        
        # Log test set statistics
        logging.info(f"\nTest Set Statistics:")
        logging.info(f"Test set size: {len(test_data)}")
        logging.info(f"Number of annotators in test set: {len(test_data['annotator_id'].unique())}")
        
        # Set data in config
        config.train_path = train_data
        config.test_path = test_data
        
        # Create trainer
        if approach == 'aart':
            trainer = SubsamplingTrainer(config, AARTModel)
        elif approach == 'multitask':
            trainer = SubsamplingTrainer(config, MultitaskModel)
        elif approach == 'annotator_embedding':
            trainer = SubsamplingTrainer(config, AnnotatorEmbeddingModel)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        # Train model
        results = trainer.train()
        
        # Save results
        results_dir = Path(f"experiments/{experiment_id}/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(results_dir / "raw_results.json", "w") as f:
            json.dump({approach: results}, f, indent=2)
        
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
    parser = argparse.ArgumentParser(description='Run experiments with different numbers of annotators')
    parser.add_argument('--approaches', nargs='+', default=['aart', 'multitask', 'annotator_embedding'],
                      help='Approaches to evaluate')
    parser.add_argument('--n_annotators', type=int, required=True,
                      help='Number of annotators to use for training')
    args = parser.parse_args()
    
    # Generate experiment ID
    experiment_id = get_experiment_id(args)
    logging.info(f"Starting experiment with ID: {experiment_id}")
    
    # Run experiments for each approach
    results = {}
    for approach in args.approaches:
        result = run_single_experiment(approach, experiment_id, args.n_annotators)
        results[approach] = result
    
    # Print final summary
    print("\n=== Final Results Summary ===")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for approach in args.approaches:
        print(f"\n=== {approach.upper()} ===")
        if results[approach] is not None:
            print(json.dumps(results[approach], indent=2))
        else:
            print("No results available")
    
    print("\n=== End of Summary ===")

if __name__ == "__main__":
    main() 