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
import pandas as pd

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
from scripts.data_loader import SentimentDataLoader
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

def run_single_experiment(approach, add_noise=False, noise_level=0.2, use_grouping=False, annotators_per_group=4, use_weighted_embeddings=False):
    """Run a single experiment with the specified approach"""
    try:
        logging.info(f"\nStarting experiment for {approach}")
        
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create config
        config = setup_config(
            approach=approach, 
            add_noise=add_noise, 
            noise_level=noise_level, 
            use_grouping=use_grouping, 
            annotators_per_group=annotators_per_group
        )
        
        # Set device in config
        config.device = device
        
        # Set weighted embeddings if requested
        if use_weighted_embeddings:
            config.use_weighted_embeddings = True
        
        # Load data
        train_path = 'data/sentiment_analysis/processed/train.json'
        test_path = 'data/sentiment_analysis/processed/test.json'
        
        # Load data into DataFrames
        train_data = pd.read_json(train_path, lines=True)
        test_data = pd.read_json(test_path, lines=True)
        
        # Set data in config
        config.train_path = train_data
        config.test_path = test_data
        
        # Create trainer
        if approach == 'majority_vote':
            trainer = Trainer(config, MajorityVoteModel)
        elif approach == 'aart':
            trainer = Trainer(config, AARTModel)
        elif approach == 'multitask':
            trainer = Trainer(config, MultitaskModel)
        elif approach == 'annotator_embedding':
            trainer = Trainer(config, AnnotatorEmbeddingModel)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        # Train and evaluate
        results = trainer.train()
        
        return results
    except Exception as e:
        logging.error(f"Error running {approach}: {str(e)}")
        traceback.print_exc()
        return None

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
                
                # Per-class metrics
                f.write("\nPer-Class Metrics:\n")
                for i in range(metrics.get('num_classes', 5)):
                    f.write(f"Class {i}:\n")
                    f.write(f"  F1: {metrics.get(f'class_{i}_f1', 'N/A'):.4f}\n")
                    f.write(f"  Precision: {metrics.get(f'class_{i}_precision', 'N/A'):.4f}\n")
                    f.write(f"  Recall: {metrics.get(f'class_{i}_recall', 'N/A'):.4f}\n")
                
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

def print_final_summary(all_results):
    """Print a summary of all results at the end of experiments."""
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    for approach, results in all_results.items():
        print(f"\n{approach.upper()}")
        print("-"*40)
        
        # Overall metrics
        print("Overall Metrics:")
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            if metric in results:
                value = results[metric]
                if isinstance(value, (int, float)):
                    print(f"  {metric.capitalize()}: {value:.4f}")
                else:
                    print(f"  {metric.capitalize()}: {value}")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        for i in range(results.get('num_classes', 5)):
            print(f"  Class {i}:")
            for metric in ['f1', 'precision', 'recall']:
                value = results.get(f'class_{i}_{metric}')
                if isinstance(value, (int, float)):
                    print(f"    {metric.capitalize()}: {value:.4f}")
                else:
                    print(f"    {metric.capitalize()}: {value}")
        
        # Annotator metrics
        print("\nAnnotator Metrics:")
        annotator_metrics = {k: v for k, v in results.items() 
                           if k.startswith('annotator_') and k.endswith('_f1')}
        for ann, score in sorted(annotator_metrics.items()):
            ann_id = ann.replace('annotator_', '').replace('_f1', '')
            if isinstance(score, (int, float)):
                print(f"  Annotator {ann_id} F1: {score:.4f}")
            else:
                print(f"  Annotator {ann_id} F1: {score}")
        
        # Aggregated annotator metrics
        print("\nAggregated Annotator Metrics:")
        mean_f1 = results.get('mean_annotator_f1')
        std_f1 = results.get('std_annotator_f1')
        if isinstance(mean_f1, (int, float)):
            print(f"  Mean Annotator F1: {mean_f1:.4f}")
        else:
            print(f"  Mean Annotator F1: {mean_f1}")
        if isinstance(std_f1, (int, float)):
            print(f"  Std Annotator F1: {std_f1:.4f}")
        else:
            print(f"  Std Annotator F1: {std_f1}")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Run sentiment analysis experiments')
    parser.add_argument('--approaches', nargs='+', required=True,
                      choices=['majority_vote', 'aart', 'multitask', 'annotator_embedding'],
                      help='Approaches to run')
    parser.add_argument('--use_weighted_embeddings', action='store_true',
                      help='Use weighted embeddings for annotator embedding model')
    parser.add_argument('--add_noise', action='store_true',
                      help='Add noise to labels during training')
    parser.add_argument('--noise_level', type=float, default=0.2,
                      help='Level of noise to add to labels (default: 0.2)')
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
                annotators_per_group=args.annotators_per_group,
                use_weighted_embeddings=args.use_weighted_embeddings
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
        
        # Print final summary
        print_final_summary(results)
    except Exception as e:
        logging.error(f"Error writing results: {str(e)}")

if __name__ == "__main__":
    main() 