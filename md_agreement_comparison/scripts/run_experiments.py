import os
from typing import Optional
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
import uuid

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
from models.implementations.aart_RINCE import AART_RINCEModel
from models.implementations.annotator_embedding import AnnotatorEmbeddingModel
from models.implementations.majority_vote import MajorityVoteModel

def setup_config(approach, add_noise=False, noise_level=0.2, use_grouping=False, annotators_per_group=4, 
                 deepspeed_config: Optional[str] = None):
    """Setup configuration for a specific approach"""
    logging.info(f"Setting up configuration for {approach}")
    
    # Create config with required approach parameter
    config = ExperimentConfig(
        approach=approach,
        add_noise=add_noise,
        noise_level=noise_level,
        use_grouping=use_grouping,
        annotators_per_group=annotators_per_group,
        deepspeed_config=deepspeed_config
    )
    
    # Set approach-specific parameters
    if approach == 'majority_vote':
        config.use_majority_vote = True
    elif approach == 'aart':
        config.lambda2 = 0.1
        config.contrastive_alpha = 0.1
    elif approach == 'aart_rince':
        config.lambda2 = 0.1
        config.contrastive_alpha = 0.1
    elif approach == 'multitask':
        pass
    elif approach == 'annotator_embedding':
        config.use_annotator_embed = True
        config.use_annotation_embed = True
    
    return config

def run_single_experiment(approach, experiment_id, add_noise=False, noise_level=0.2, use_grouping=False, annotators_per_group=4, use_weighted_embeddings=False, 
                          deepspeed_config: Optional[str] = None):
    """Run a single experiment with the specified approach"""
    try:
        logging.info(f"\nStarting experiment for {approach}")
        
        # Create config
        config = setup_config(
            approach=approach, 
            add_noise=add_noise, 
            noise_level=noise_level, 
            use_grouping=use_grouping, 
            annotators_per_group=annotators_per_group,
            deepspeed_config=deepspeed_config
        )
        
        # Set weighted embeddings if requested
        if use_weighted_embeddings:
            config.use_weighted_embeddings = True
            
        # Set experiment ID and directories
        config.experiment_id = experiment_id
        config.checkpoint_dir = Path(f"experiments/{experiment_id}/models/checkpoints/{approach}")
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set data paths in config
        config.train_path = 'data/md_agreement/processed/train.json'
        config.test_path = 'data/md_agreement/processed/test.json'
        
        # Create trainer
        if approach == 'majority_vote':
            trainer = Trainer(config, MajorityVoteModel)
        elif approach == 'aart':
            trainer = Trainer(config, AARTModel)
        elif approach == 'aart_rince':
            trainer = Trainer(config, AART_RINCEModel)
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

def write_final_comparison(results, output_path):
    """Write comparison of results with detailed metrics."""
    def fmt(val):
        return f"{val:.4f}" if isinstance(val, (float, int)) else str(val)

    logging.info("Writing final comparison…")
    with open(output_path, 'w') as f:
        f.write("=== Final Comparison of All Approaches ===\n")
        f.write(f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

        for approach in ['aart', 'aart_rince', 'multitask', 'annotator_embedding', 'majority_vote']:
            res = results.get(approach)
            f.write(f"\n=== {approach.upper()} ===\n")
            if not isinstance(res, dict):
                f.write("No results available\n")
                continue

            # Overall
            f.write("\nOverall Metrics:\n")
            f.write(f"  Accuracy:  {fmt(res.get('accuracy','N/A'))}\n")
            f.write(f"  F1 Score:  {fmt(res.get('f1','N/A'))}\n")
            f.write(f"  Precision: {fmt(res.get('precision','N/A'))}\n")
            f.write(f"  Recall:    {fmt(res.get('recall','N/A'))}\n")

            # Per-class
            nc = res.get('num_classes', 2)
            try: nc = int(nc)
            except: nc = 2
            f.write("\nPer-Class Metrics:\n")
            for i in range(nc):
                f.write(f"  Class {i}:\n")
                f.write(f"    F1:        {fmt(res.get(f'class_{i}_f1','N/A'))}\n")
                f.write(f"    Precision:{fmt(res.get(f'class_{i}_precision','N/A'))}\n")
                f.write(f"    Recall:   {fmt(res.get(f'class_{i}_recall','N/A'))}\n")

            # Annotator-level
            f.write("\nAnnotator Metrics:\n")
            f.write(f"  Mean Annotator F1: {fmt(res.get('mean_annotator_f1','N/A'))}\n")
            f.write(f"  Std Annotator F1:  {fmt(res.get('std_annotator_f1','N/A'))}\n")
            f.write(f"  Min Annotator F1:  {fmt(res.get('min_annotator_f1','N/A'))}\n")
            f.write(f"  Max Annotator F1:  {fmt(res.get('max_annotator_f1','N/A'))}\n")
            f.write(f"  Num Annotators:    {fmt(res.get('num_annotators_evaluated','N/A'))}\n")

            # Individual annotators
            per = res.get('per_annotator_metrics', {})
            if isinstance(per, dict):
                f.write("\nIndividual Annotator Metrics:\n")
                for aid, m in per.items():
                    f.write(
                        f"  Annotator {aid}: "
                        f"F1={fmt(m.get('f1','N/A'))}, "
                        f"Acc={fmt(m.get('accuracy','N/A'))}, "
                        f"Samples={fmt(m.get('num_samples','N/A'))}\n"
                    )

        f.write("\n=== End of Report ===\n")


def print_final_summary(results):
    """Print a summary of the results to the console, safely formatting missing values."""
    def fmt(val):
        return f"{val:.4f}" if isinstance(val, (float, int)) else str(val)

    print("\n=== Final Results Summary ===")
    print(f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    for approach in ['aart', 'aart_rince', 'multitask', 'annotator_embedding', 'majority_vote']:
        res = results.get(approach)
        print(f"\n=== {approach.upper()} ===")
        if not isinstance(res, dict):
            print("No results available")
            continue

        # Overall
        print("\nOverall Metrics:")
        print(f"  Accuracy:  {fmt(res.get('accuracy','N/A'))}")
        print(f"  F1 Score:  {fmt(res.get('f1','N/A'))}")
        print(f"  Precision: {fmt(res.get('precision','N/A'))}")
        print(f"  Recall:    {fmt(res.get('recall','N/A'))}")

        # Per-class
        nc = res.get('num_classes', 2)
        try: nc = int(nc)
        except: nc = 2
        print("\nPer-Class Metrics:")
        for i in range(nc):
            print(f"  Class {i}:")
            print(f"    F1:        {fmt(res.get(f'class_{i}_f1','N/A'))}")
            print(f"    Precision:{fmt(res.get(f'class_{i}_precision','N/A'))}")
            print(f"    Recall:   {fmt(res.get(f'class_{i}_recall','N/A'))}")

        # Annotator
        print("\nAnnotator Metrics:")
        print(f"  Mean Annotator F1: {fmt(res.get('mean_annotator_f1','N/A'))}")
        print(f"  Std Annotator F1:  {fmt(res.get('std_annotator_f1','N/A'))}")
        print(f"  Min Annotator F1:  {fmt(res.get('min_annotator_f1','N/A'))}")
        print(f"  Max Annotator F1:  {fmt(res.get('max_annotator_f1','N/A'))}")
        print(f"  Num Annotators:    {fmt(res.get('num_annotators_evaluated','N/A'))}")

        # Individual breakdown
        per = res.get('per_annotator_metrics', {})
        if isinstance(per, dict):
            print("\nIndividual Annotator Metrics:")
            for aid, m in per.items():
                print(
                    f"  Annotator {aid}: "
                    f"F1={fmt(m.get('f1','N/A'))}, "
                    f"Acc={fmt(m.get('accuracy','N/A'))}, "
                    f"Samples={fmt(m.get('num_samples','N/A'))}"
                )
    print("\n=== End of Summary ===\n")

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_experiment_id(args):
    """Generate a unique experiment ID based on parameters"""
    # Create a unique ID
    unique_id = str(uuid.uuid4())[:8]
    
    # Create a descriptive name
    name_parts = []
    
    # Add approaches
    approach_str = '_'.join(sorted(args.approaches))
    name_parts.append(f"approaches-{approach_str}")
    
    # Add grouping info if enabled
    if args.use_grouping:
        name_parts.append(f"group-{args.annotators_per_group}")
    
    # Add noise info if enabled
    if args.add_noise:
        name_parts.append(f"noise-{args.noise_level}")
    
    # Add weighted embeddings info if enabled
    if args.use_weighted_embeddings:
        name_parts.append("weighted")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts.append(timestamp)
    
    # Combine with unique ID
    experiment_name = "__".join(name_parts)
    experiment_id = f"{experiment_name}_{unique_id}"
    
    return experiment_id

def main():
    parser = argparse.ArgumentParser(description='Run MD agreement experiments')
    parser.add_argument('--approaches', nargs='+', required=True,
                      choices=['majority_vote', 'aart', 'aart_rince', 'multitask', 'annotator_embedding'],
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
    parser.add_argument('--experiment_id', type=str, default=None,
                      help='Optional experiment ID to use (if not provided, a new one will be generated)')
    
    #Add deepspeed config
    parser.add_argument('--deepspeed_config', type=str, default=None,
                    help='Path to DeepSpeed JSON config')
    parser.add_argument('--local_rank', type=int, default=0,
                    help='DeepSpeed local rank (do not set)')
    
    # Parse the arguments before using them
    args = parser.parse_args()

    # Tell the rest of the code which GPU this process should use
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    # Generate unique experiment ID
    experiment_id = args.experiment_id if args.experiment_id else get_experiment_id(args)
    
    # Create experiment-specific directories
    experiment_dir = Path("experiments") / experiment_id
    results_dir = experiment_dir / "results"
    logs_dir = experiment_dir / "logs"
    
    # Create all directories
    for dir_path in [results_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize logging with experiment-specific log file
    log_file = logs_dir / 'experiment.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log experiment details
    logging.info(f"Starting experiment: {experiment_id}")
    logging.info(f"Selected approaches: {args.approaches}")
    logging.info(f"Results will be saved in: {experiment_dir}")
    
    # Save experiment configuration
    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        config = {
            'experiment_id': experiment_id,
            'approaches': args.approaches,
            'use_weighted_embeddings': args.use_weighted_embeddings,
            'add_noise': args.add_noise,
            'noise_level': args.noise_level,
            'use_grouping': args.use_grouping,
            'annotators_per_group': args.annotators_per_group,
            'timestamp': datetime.now().isoformat()
        }
        json.dump(config, f, indent=2)
    
    # Set random seeds for reproducibility
    set_seeds()
    
    # Run experiments
    results = {}
    for approach in args.approaches:
        try:
            logging.info(f"\nStarting experiment for {approach}")
            # Pass experiment ID to run_single_experiment
            results[approach] = run_single_experiment(
                approach, 
                experiment_id=experiment_id,
                add_noise=args.add_noise, 
                noise_level=args.noise_level,
                use_grouping=args.use_grouping,
                annotators_per_group=args.annotators_per_group,
                use_weighted_embeddings=args.use_weighted_embeddings,
                deepspeed_config=args.deepspeed_config
            )
        except Exception as e:
            logging.error(f"Error running {approach}: {str(e)}")
            results[approach] = {"error": str(e)}
    
    try:
        # Write final comparison to experiment-specific directory
        comparison_path = results_dir / "final_comparison.txt"
        write_final_comparison(results, comparison_path)
        
        # Save raw results
        raw_results_path = results_dir / "raw_results.json"
        with open(raw_results_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = {k: (v if isinstance(v, dict) else {"error": str(v)}) 
                                  for k, v in results.items()}
            json.dump(serializable_results, f, indent=2)
        
        # Create a summary file with key metrics
        summary_path = results_dir / "summary.json"
        summary = {
            'experiment_id': experiment_id,
            'approaches': args.approaches,
            'metrics': {}
        }
        
        for approach in args.approaches:
            if approach in results and isinstance(results[approach], dict):
                metrics = results[approach]
                summary['metrics'][approach] = {
                    'accuracy': metrics.get('accuracy', 'N/A'),
                    'f1': metrics.get('f1', 'N/A'),
                    'precision': metrics.get('precision', 'N/A'),
                    'recall': metrics.get('recall', 'N/A'),
                    'mean_annotator_f1': metrics.get('mean_annotator_f1', 'N/A')
                }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"\nExperiments completed! Results saved in:")
        logging.info(f"- {raw_results_path}")
        logging.info(f"- {comparison_path}")
        logging.info(f"- {summary_path}")
        
        # Print final summary
        print_final_summary(results)
    except Exception as e:
        logging.error(f"Error writing results: {str(e)}")

if __name__ == "__main__":
    main()