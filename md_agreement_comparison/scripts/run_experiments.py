import os
import torch
import json
from pathlib import Path
from datetime import datetime
import argparse
import wandb
import sys
import logging
import traceback
import numpy as np

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log'),
        logging.StreamHandler()
    ]
)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Update imports to be absolute from project root
from scripts.config import ExperimentConfig
from scripts.train import Trainer
from scripts.metrics import evaluate_model
from scripts.data_loader import MDAgreementDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Import models
from models.implementations.multitask import MultitaskModel
from models.implementations.aart import AARTModel
from models.implementations.annotator_embedding import AnnotatorEmbeddingModel
from models.implementations.majority_vote import MajorityVoteModel

def setup_config(approach, add_noise=False, noise_level=0.2, use_grouping=False, annotators_per_group=4):
    """Setup configuration for a specific approach"""
    logging.info(f"Setting up configuration for {approach}")
    
    # Create config with required approach parameter
    config = ExperimentConfig(
        approach=approach,
        add_noise=add_noise,
        noise_level=noise_level,
        use_grouping=use_grouping,
        annotators_per_group=annotators_per_group
    )
    
    # Set approach-specific parameters
    if approach == 'majority_vote':
        config.use_majority_vote = True
    elif approach == 'aart':
        config.lambda2 = 0.1
        config.contrastive_alpha = 0.1
    elif approach == 'multitask':
        pass
    elif approach == 'annotator_embedding':
        config.use_annotator_embed = True
        config.use_annotation_embed = True
    
    return config

def run_single_experiment(approach, add_noise=False, noise_level=0.2, use_grouping=False, annotators_per_group=4):
    logging.info(f"\n=== Running {approach.upper()} Experiment ===")
    
    try:
        # Setup configuration with noise parameters
        config = setup_config(approach, add_noise, noise_level, use_grouping, annotators_per_group)
        
        # Determine model class based on approach
        model_class = {
            'majority_vote': MajorityVoteModel,
            'aart': AARTModel,
            'multitask': MultitaskModel,
            'annotator_embedding': AnnotatorEmbeddingModel
        }[approach]
        
        # Check if data exists
        data_path = Path("data/md_agreement/processed")
        if not data_path.exists() or not list(data_path.glob("*.json")):
            logging.error("Data not found. Please run download_data.py first")
            raise FileNotFoundError("Dataset files not found")
        
        # Train and evaluate
        trainer = Trainer(config, model_class)
        metrics = trainer.train()  # This now returns the evaluation metrics
        
        # Log metrics
        logging.info(f"\nMetrics for {approach}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f"{metric}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error in {approach} experiment: {str(e)}")
        traceback.print_exc()
        return {
            'error': str(e),
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'mean_annotator_f1': 0.0,
            'std_annotator_f1': 0.0
        }

def write_final_comparison(results):
    """Write comparison of results with detailed metrics"""
    logging.info("Writing final comparison...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "final_comparison.txt", 'w') as f:
        f.write("=== Final Comparison of All Approaches ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        approaches = ['aart', 'multitask', 'annotator_embedding']
        
        for approach in approaches:
            f.write(f"\n=== {approach.upper()} ===\n")
            if approach in results and isinstance(results[approach], dict):
                metrics = results[approach]
                
                # Overall metrics
                f.write("\nOverall Metrics:\n")
                f.write(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n")
                f.write(f"F1 Score: {metrics.get('f1', 'N/A'):.4f}\n")
                f.write(f"Precision: {metrics.get('precision', 'N/A'):.4f}\n")
                f.write(f"Recall: {metrics.get('recall', 'N/A'):.4f}\n")
                
                # Annotator metrics
                f.write("\nAnnotator Metrics:\n")
                f.write(f"Mean Annotator F1: {metrics.get('mean_annotator_f1', 'N/A'):.4f}\n")
                f.write(f"Std Annotator F1: {metrics.get('std_annotator_f1', 'N/A'):.4f}\n")
                f.write(f"Min Annotator F1: {metrics.get('min_annotator_f1', 'N/A'):.4f}\n")
                f.write(f"Max Annotator F1: {metrics.get('max_annotator_f1', 'N/A'):.4f}\n")
                
                # Save detailed per-annotator metrics to separate file
                if 'per_annotator_metrics' in metrics:
                    annotator_file = results_dir / f"{approach}_annotator_metrics.json"
                    with open(annotator_file, 'w') as af:
                        json.dump(metrics['per_annotator_metrics'], af, indent=2)
                    f.write(f"\nDetailed per-annotator metrics saved to: {annotator_file}\n")
            else:
                f.write("No results available\n")
        
        f.write("\n=== End of Report ===\n")

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approaches', nargs='+', 
                      default=['majority_vote', 'aart', 'multitask', 'annotator_embedding'],
                      help='Which approaches to run')
    parser.add_argument('--add_noise', action='store_true',
                      help='Add annotator-specific noise to labels')
    parser.add_argument('--noise_level', type=float, default=0.2,
                      help='Noise level to apply to all annotators (0.0 to 1.0)')
    parser.add_argument('--use_weighted_embeddings', 
                       action='store_true',
                       help='Whether to use weighted embeddings for annotator representations')
    parser.add_argument('--use_grouping', action='store_true',
                      help='Enable annotator grouping')
    parser.add_argument('--annotators_per_group', type=int, default=4,
                      help='Number of annotators per group when grouping is enabled')
    
    # Parse the arguments before using them
    args = parser.parse_args()
    
    logging.info("Starting experiments")
    logging.info(f"Selected approaches: {args.approaches}")
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    set_seeds()
    
    # Run experiments
    results = {}
    for approach in args.approaches:
        try:
            logging.info(f"\nStarting experiment for {approach}")
            # Pass noise parameters to run_single_experiment
            results[approach] = run_single_experiment(
                approach, 
                add_noise=args.add_noise, 
                noise_level=args.noise_level,
                use_grouping=args.use_grouping,
                annotators_per_group=args.annotators_per_group
            )
        except Exception as e:
            logging.error(f"Error running {approach}: {str(e)}")
            results[approach] = {"error": str(e)}
    
    try:
        # Write final comparison
        write_final_comparison(results)
        
        # Save raw results
        raw_results_path = Path("results/raw_results.json")
        with open(raw_results_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = {k: (v if isinstance(v, dict) else {"error": str(v)}) 
                                  for k, v in results.items()}
            json.dump(serializable_results, f, indent=2)
        
        logging.info(f"\nExperiments completed! Results saved in:")
        logging.info(f"- {raw_results_path}")
        logging.info(f"- results/final_comparison.txt")
    except Exception as e:
        logging.error(f"Error writing results: {str(e)}")

if __name__ == "__main__":
    main()