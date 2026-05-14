import os
import torch
import json
import csv
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
import random
from typing import List, Dict, Any, Optional
from scipy import stats
import itertools

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
from models.implementations.aart_Rince_new import NewRinceModel
from models.implementations.annotator_embedding import AnnotatorEmbeddingModel
from models.implementations.majority_vote import MajorityVoteModel
from models.implementations.annotator_embedding_rince import AnnotatorEmbeddingRinceModel

def setup_config(approach, add_noise=False, noise_level=0.2, noise_strategy='fixed',
                renegade_percent=0.1, renegade_flip_prob=0.7, use_grouping=False,
                annotators_per_group=4, gamma=0.5, confusion_seed=42,
                embeddings_path=None, **kwargs):
    """Setup configuration for a specific approach"""
    logging.info(f"Setting up configuration for {approach}")
    
    # Create config with required approach parameter
    config = ExperimentConfig(
        approach=approach,
        add_noise=add_noise,
        noise_level=noise_level,
        noise_strategy=noise_strategy,
        renegade_percent=renegade_percent,
        renegade_flip_prob=renegade_flip_prob,
        use_grouping=use_grouping,
        annotators_per_group=annotators_per_group,
        gamma=gamma,
        confusion_seed=confusion_seed,
        embeddings_path=embeddings_path,
    )
    
    # Set approach-specific parameters
    if approach == 'majority_vote':
        config.use_majority_vote = True
    elif approach == 'aart':
        config.lambda2 = kwargs.get('lambda2', 0.1)
        config.contrastive_alpha = kwargs.get('contrastive_alpha', 0.1)
    elif approach == 'aart_rince':
        config.lambda2 = kwargs.get('lambda2', 0.1)
        config.contrastive_alpha = kwargs.get('contrastive_alpha', 0.1)
        config.temperature = kwargs.get('temperature', 0.07)
        config.rince_lambda = kwargs.get('rince_lambda', 0.5)
        config.rince_q = kwargs.get('rince_q', 0.5)
    elif approach == 'multitask':
        pass
    elif approach == 'annotator_embedding':
        config.use_annotator_embed = True
        config.use_annotation_embed = True
    elif approach == 'annotator_embedding_rince':
        config.use_annotator_embed = True
        config.use_annotation_embed = True
        config.lambda2 = kwargs.get('lambda2', 0.1)
        config.temperature = kwargs.get('temperature', 0.07)
        config.rince_lambda = kwargs.get('rince_lambda', 1.0)
        config.rince_q = kwargs.get('rince_q', 1.0)
    
    return config

def run_single_experiment(approach, experiment_id, seed, run_number, add_noise=False,
                         noise_level=0.2, noise_strategy='fixed', renegade_percent=0.1,
                         renegade_flip_prob=0.7, use_grouping=False, annotators_per_group=4,
                         use_weighted_embeddings=False, num_epochs=None,
                         gamma=0.5, confusion_seed=42, embeddings_path=None, **hyperparams):
    """Run a single experiment with the specified approach and seed"""
    try:
        logging.info(f"Starting experiment for {approach} (seed={seed}, run={run_number})")
        
        # Set seeds for reproducibility
        set_seeds(seed)
        
        # Create config
        config = setup_config(
            approach=approach,
            add_noise=add_noise,
            noise_level=noise_level,
            noise_strategy=noise_strategy,
            renegade_percent=renegade_percent,
            renegade_flip_prob=renegade_flip_prob,
            use_grouping=use_grouping,
            annotators_per_group=annotators_per_group,
            gamma=gamma,
            confusion_seed=confusion_seed,
            embeddings_path=embeddings_path,
            **hyperparams
        )
        
        # Set weighted embeddings if requested
        if use_weighted_embeddings:
            config.use_weighted_embeddings = True
        
        # Override num_epochs if specified
        if num_epochs is not None:
            config.num_epochs = num_epochs
        
        # Set seed in config for use in Trainer
        config.seed = seed
            
        # Set experiment ID and directories
        config.experiment_id = experiment_id
        config.checkpoint_dir = Path(f"multi_seed_experiments/{experiment_id}/models/checkpoints/{approach}/seed_{seed}/run_{run_number}")
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set data paths in config
        config.train_path = 'data/epic_dataset/processed/train.json'
        config.test_path = 'data/epic_dataset/processed/test.json'
        
        # Create trainer
        if approach == 'majority_vote':
            trainer = Trainer(config, MajorityVoteModel)
        elif approach == 'aart':
            trainer = Trainer(config, AARTModel)
        elif approach == 'aart_rince':
            trainer = Trainer(config, NewRinceModel)
        elif approach == 'multitask':
            trainer = Trainer(config, MultitaskModel)
        elif approach == 'annotator_embedding':
            trainer = Trainer(config, AnnotatorEmbeddingModel)
        elif approach == 'annotator_embedding_rince':
            trainer = Trainer(config, AnnotatorEmbeddingRinceModel)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        # Train and evaluate (train() already evaluates and returns metrics)
        results = trainer.train()
        
        logging.info(f"Training completed for {approach} (seed={seed}, run={run_number})")
        
        return results
    except Exception as e:
        logging.error(f"Error running {approach} (seed={seed}, run={run_number}): {str(e)}")
        traceback.print_exc()
        return None

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    
    # PyTorch deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # CUDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)

def extract_key_metrics(results):
    """Extract key metrics from results for statistical analysis"""
    if not results or not isinstance(results, dict):
        return {}
    
    key_metrics = {
        'accuracy': results.get('accuracy', None),
        'f1': results.get('f1', None),
        'precision': results.get('precision', None),
        'recall': results.get('recall', None),
        'mean_annotator_f1': results.get('mean_annotator_f1', None),
        'std_annotator_f1': results.get('std_annotator_f1', None),
        'min_annotator_f1': results.get('min_annotator_f1', None),
        'max_annotator_f1': results.get('max_annotator_f1', None),
        'num_annotators_evaluated': results.get('num_annotators_evaluated', None)
    }
    
    # Add per-class metrics if available
    num_classes = results.get('num_classes', 2)
    for i in range(num_classes):
        key_metrics[f'class_{i}_f1'] = results.get(f'class_{i}_f1', None)
        key_metrics[f'class_{i}_precision'] = results.get(f'class_{i}_precision', None)
        key_metrics[f'class_{i}_recall'] = results.get(f'class_{i}_recall', None)
    
    return key_metrics

def calculate_statistics(metric_values):
    """Calculate statistics for a list of metric values"""
    if not metric_values or all(v is None for v in metric_values):
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0,
            'ci_95_lower': None,
            'ci_95_upper': None
        }
    
    # Filter out None values
    valid_values = [v for v in metric_values if v is not None]
    
    if len(valid_values) == 0:
        return {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0,
            'ci_95_lower': None,
            'ci_95_upper': None
        }
    
    mean_val = np.mean(valid_values)
    std_val = np.std(valid_values, ddof=1)  # Sample standard deviation
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)
    
    # Calculate 95% confidence interval
    if len(valid_values) > 1:
        ci_95 = stats.t.interval(0.95, len(valid_values)-1, 
                                loc=mean_val, scale=stats.sem(valid_values))
        ci_95_lower, ci_95_upper = ci_95
    else:
        ci_95_lower, ci_95_upper = mean_val, mean_val
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'count': len(valid_values),
        'ci_95_lower': ci_95_lower,
        'ci_95_upper': ci_95_upper
    }

def write_csv_results(all_results, output_path):
    """Write results to CSV format for easy analysis"""
    logging.info(f"Writing CSV results to {output_path}")
    
    # Prepare data for CSV
    csv_data = []
    
    for experiment_config_tuple, results in all_results.items():
        # Convert tuple back to dictionary
        experiment_config = dict(experiment_config_tuple)
        
        for approach, approach_results in results.items():
            for seed, seed_results in approach_results.items():
                for run_number, run_results in seed_results.items():
                    if run_results is not None:
                        metrics = extract_key_metrics(run_results)
                        
                        row = {
                            'experiment_id': experiment_config['experiment_id'],
                            'approach': approach,
                            'seed': seed,
                            'run_number': run_number,
                            'add_noise': experiment_config['add_noise'],
                            'noise_level': experiment_config['noise_level'],
                            'noise_strategy': experiment_config['noise_strategy'],
                            'renegade_percent': experiment_config['renegade_percent'],
                            'renegade_flip_prob': experiment_config['renegade_flip_prob'],
                            'use_grouping': experiment_config['use_grouping'],
                            'annotators_per_group': experiment_config['annotators_per_group'],
                            'use_weighted_embeddings': experiment_config['use_weighted_embeddings'],
                            **metrics
                        }
                        csv_data.append(row)
    
    # Write to CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        logging.info(f"CSV results written with {len(csv_data)} rows")
    else:
        logging.warning("No data to write to CSV")

def write_statistical_summary(all_results, output_path):
    """Write statistical summary across seeds and runs"""
    logging.info(f"Writing statistical summary to {output_path}")
    
    summary_data = []
    
    for experiment_config_tuple, results in all_results.items():
        # Convert tuple back to dictionary
        experiment_config = dict(experiment_config_tuple)
        
        for approach, approach_results in results.items():
            # Collect all metric values across seeds and runs
            metric_aggregates = {}
            
            for seed, seed_results in approach_results.items():
                for run_number, run_results in seed_results.items():
                    if run_results is not None:
                        metrics = extract_key_metrics(run_results)
                        
                        for metric_name, metric_value in metrics.items():
                            if metric_name not in metric_aggregates:
                                metric_aggregates[metric_name] = []
                            if metric_value is not None:
                                metric_aggregates[metric_name].append(metric_value)
            
            # Calculate statistics for each metric
            for metric_name, metric_values in metric_aggregates.items():
                stats_result = calculate_statistics(metric_values)
                
                summary_row = {
                    'experiment_id': experiment_config['experiment_id'],
                    'approach': approach,
                    'metric': metric_name,
                    'mean': stats_result['mean'],
                    'std': stats_result['std'],
                    'min': stats_result['min'],
                    'max': stats_result['max'],
                    'count': stats_result['count'],
                    'ci_95_lower': stats_result['ci_95_lower'],
                    'ci_95_upper': stats_result['ci_95_upper'],
                    'add_noise': experiment_config['add_noise'],
                    'noise_level': experiment_config['noise_level'],
                    'noise_strategy': experiment_config['noise_strategy'],
                    'renegade_percent': experiment_config['renegade_percent'],
                    'renegade_flip_prob': experiment_config['renegade_flip_prob'],
                    'use_grouping': experiment_config['use_grouping'],
                    'annotators_per_group': experiment_config['annotators_per_group'],
                    'use_weighted_embeddings': experiment_config['use_weighted_embeddings']
                }
                summary_data.append(summary_row)
    
    # Write to CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        logging.info(f"Statistical summary written with {len(summary_data)} rows")
    else:
        logging.warning("No data to write to statistical summary")

def get_experiment_id(args):
    """Generate a unique experiment ID based on parameters"""
    unique_id = str(uuid.uuid4())[:8]
    
    name_parts = []
    
    # Add dataset name to make IDs unique across datasets
    name_parts.append("epic_dataset")
    
    # Add approaches
    approach_str = '_'.join(sorted(args.approaches))
    name_parts.append(f"approaches-{approach_str}")
    
    # Add seeds info
    seed_str = '_'.join(map(str, args.seeds))
    name_parts.append(f"seeds-{seed_str}")
    
    # Add runs per seed
    name_parts.append(f"runs-{args.runs_per_seed}")
    
    # Add grouping info if enabled
    if args.use_grouping:
        name_parts.append(f"grouping-{args.annotators_per_group}")
    
    # Add noise info if enabled
    if args.add_noise:
        if args.noise_strategy == 'renegade':
            name_parts.append(f"renegade-{args.renegade_percent}-{args.renegade_flip_prob}")
        elif args.noise_strategy == 'instance_dependent':
            name_parts.append(f"instance_dep-{args.noise_level}")
        elif args.noise_strategy == 'combined':
            name_parts.append(f"combined-{args.noise_level}-g{args.gamma}")
        else:
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
    parser = argparse.ArgumentParser(description='Run EPIC dataset experiments with multiple seeds')
    
    # Core experiment parameters
    parser.add_argument('--approaches', nargs='+', required=True,
                      choices=['majority_vote', 'aart', 'aart_rince', 'multitask', 
                              'annotator_embedding', 'annotator_embedding_rince'],
                      help='Approaches to run')
    parser.add_argument('--seeds', nargs='+', type=int, required=True,
                      help='List of seeds to run experiments on')
    parser.add_argument('--runs_per_seed', type=int, default=1,
                      help='Number of runs per seed (default: 1)')
    
    # Noise parameters
    parser.add_argument('--add_noise', action='store_true',
                      help='Add noise to labels during training')
    parser.add_argument('--noise_level', type=float, default=0.2,
                      help='Level of noise to add to labels (default: 0.2)')
    parser.add_argument('--noise_strategy', type=str, default='fixed',
                      choices=['fixed', 'random', 'custom', 'renegade',
                               'instance_dependent', 'combined'],
                      help='Strategy for adding noise (default: fixed)')
    parser.add_argument('--gamma', type=float, default=0.5,
                      help='Instance confusion scaling for combined strategy (default: 0.5)')
    parser.add_argument('--confusion_seed', type=int, default=42,
                      help='Fixed seed for global confusion vector w (default: 42)')
    parser.add_argument('--embeddings_path', type=str, default=None,
                      help='Path to precomputed RoBERTa [CLS] embeddings .npy file')
    parser.add_argument('--renegade_percent', type=float, default=0.1,
                      help='Percentage of annotators to be renegades (default: 0.1)')
    parser.add_argument('--renegade_flip_prob', type=float, default=0.7,
                      help='Probability of flipping labels for renegade annotators (default: 0.7)')
    
    # Grouping parameters
    parser.add_argument('--use_grouping', action='store_true',
                      help='Enable annotator grouping')
    parser.add_argument('--annotators_per_group', type=int, default=4,
                      help='Number of annotators per group when grouping is enabled')
    
    # Model-specific parameters
    parser.add_argument('--use_weighted_embeddings', action='store_true',
                      help='Use weighted embeddings for annotator embedding model')
    
    # Hyperparameter overrides
    parser.add_argument('--lambda2', type=float, default=None,
                      help='Override lambda2 hyperparameter')
    parser.add_argument('--contrastive_alpha', type=float, default=None,
                      help='Override contrastive_alpha hyperparameter')
    parser.add_argument('--temperature', type=float, default=None,
                      help='Override temperature hyperparameter')
    parser.add_argument('--rince_lambda', type=float, default=None,
                      help='Override rince_lambda hyperparameter')
    parser.add_argument('--rince_q', type=float, default=None,
                      help='Override rince_q hyperparameter')
    
    # Output parameters
    parser.add_argument('--num_epochs', type=int, default=None,
                      help='Number of training epochs (overrides config default)')
    
    parser.add_argument('--experiment_id', type=str, default=None,
                      help='Optional experiment ID to use (if not provided, a new one will be generated)')
    parser.add_argument('--output_dir', type=str, default='multi_seed_experiments',
                      help='Base directory for experiment outputs (default: multi_seed_experiments)')
    
    args = parser.parse_args()
    
    # Generate unique experiment ID
    experiment_id = args.experiment_id if args.experiment_id else get_experiment_id(args)
    
    # Create experiment-specific directories
    base_dir = Path(args.output_dir)
    experiment_dir = base_dir / experiment_id
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
    logging.info(f"Starting multi-seed experiment: {experiment_id}")
    logging.info(f"Selected approaches: {args.approaches}")
    logging.info(f"Seeds: {args.seeds}")
    logging.info(f"Runs per seed: {args.runs_per_seed}")
    logging.info(f"Total experiments: {len(args.approaches) * len(args.seeds) * args.runs_per_seed}")
    logging.info(f"Results will be saved in: {experiment_dir}")
    
    # Save experiment configuration
    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        config = {
            'experiment_id': experiment_id,
            'approaches': args.approaches,
            'seeds': args.seeds,
            'runs_per_seed': args.runs_per_seed,
            'use_weighted_embeddings': args.use_weighted_embeddings,
            'add_noise': args.add_noise,
            'noise_level': args.noise_level,
            'noise_strategy': args.noise_strategy,
            'renegade_percent': args.renegade_percent,
            'renegade_flip_prob': args.renegade_flip_prob,
            'use_grouping': args.use_grouping,
            'annotators_per_group': args.annotators_per_group,
            'hyperparameters': {
                'lambda2': args.lambda2,
                'contrastive_alpha': args.contrastive_alpha,
                'temperature': args.temperature,
                'rince_lambda': args.rince_lambda,
                'rince_q': args.rince_q
            },
            'timestamp': datetime.now().isoformat()
        }
        json.dump(config, f, indent=2)
    
    # Prepare hyperparameters
    hyperparams = {}
    if args.lambda2 is not None:
        hyperparams['lambda2'] = args.lambda2
    if args.contrastive_alpha is not None:
        hyperparams['contrastive_alpha'] = args.contrastive_alpha
    if args.temperature is not None:
        hyperparams['temperature'] = args.temperature
    if args.rince_lambda is not None:
        hyperparams['rince_lambda'] = args.rince_lambda
    if args.rince_q is not None:
        hyperparams['rince_q'] = args.rince_q
    
    # Run experiments
    all_results = {}
    experiment_config = {
        'experiment_id': experiment_id,
        'add_noise': args.add_noise,
        'noise_level': args.noise_level,
        'noise_strategy': args.noise_strategy,
        'renegade_percent': args.renegade_percent,
        'renegade_flip_prob': args.renegade_flip_prob,
        'use_grouping': args.use_grouping,
        'annotators_per_group': args.annotators_per_group,
        'use_weighted_embeddings': args.use_weighted_embeddings
    }
    
    all_results[tuple(experiment_config.items())] = {}
    
    total_experiments = len(args.approaches) * len(args.seeds) * args.runs_per_seed
    completed_experiments = 0
    
    for approach in args.approaches:
        logging.info(f"\nStarting experiments for approach: {approach}")
        all_results[tuple(experiment_config.items())][approach] = {}
        
        for seed in args.seeds:
            logging.info(f"  Starting experiments for seed: {seed}")
            all_results[tuple(experiment_config.items())][approach][seed] = {}
            
            for run_number in range(1, args.runs_per_seed + 1):
                try:
                    logging.info(f"    Running {approach} (seed={seed}, run={run_number})")
                    
                    results = run_single_experiment(
                        approach=approach,
                        experiment_id=experiment_id,
                        seed=seed,
                        run_number=run_number,
                        add_noise=args.add_noise,
                        noise_level=args.noise_level,
                        noise_strategy=args.noise_strategy,
                        renegade_percent=args.renegade_percent,
                        renegade_flip_prob=args.renegade_flip_prob,
                        use_grouping=args.use_grouping,
                        annotators_per_group=args.annotators_per_group,
                        use_weighted_embeddings=args.use_weighted_embeddings,
                        num_epochs=args.num_epochs,
                        gamma=args.gamma,
                        confusion_seed=args.confusion_seed,
                        embeddings_path=args.embeddings_path,
                        **hyperparams
                    )
                    
                    all_results[tuple(experiment_config.items())][approach][seed][run_number] = results
                    completed_experiments += 1
                    
                    logging.info(f"    Completed {approach} (seed={seed}, run={run_number}) - {completed_experiments}/{total_experiments}")
                    
                except Exception as e:
                    logging.error(f"    Error running {approach} (seed={seed}, run={run_number}): {str(e)}")
                    all_results[tuple(experiment_config.items())][approach][seed][run_number] = None
                    completed_experiments += 1
    
    # Write results
    try:
        # Write CSV results
        csv_path = results_dir / "results.csv"
        write_csv_results(all_results, csv_path)
        
        # Write statistical summary
        stats_path = results_dir / "statistical_summary.csv"
        write_statistical_summary(all_results, stats_path)
        
        # Save raw results
        raw_results_path = results_dir / "raw_results.json"
        with open(raw_results_path, 'w') as f:
            # Convert tuple keys to strings for JSON serialization
            json_results = {}
            for config_key, config_results in all_results.items():
                config_str = str(config_key)
                json_results[config_str] = config_results
            json.dump(json_results, f, indent=2)
        
        logging.info(f"\nExperiments completed! Results saved in:")
        logging.info(f"- {csv_path}")
        logging.info(f"- {stats_path}")
        logging.info(f"- {raw_results_path}")
        
        # Print final summary
        print_final_summary(all_results)
        
    except Exception as e:
        logging.error(f"Error writing results: {str(e)}")
        traceback.print_exc()

def print_final_summary(all_results):
    """Print a summary of all experiment results"""
    print("\n=== Final Results Summary ===")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for experiment_config_tuple, results in all_results.items():
        # Convert tuple back to dictionary
        experiment_config = dict(experiment_config_tuple)
        print(f"Experiment Config: {experiment_config}")
        
        for approach, approach_results in results.items():
            print(f"\n=== {approach.upper()} ===")
            
            # Collect all metric values across seeds and runs
            metric_aggregates = {}
            
            for seed, seed_results in approach_results.items():
                for run_number, run_results in seed_results.items():
                    if run_results is not None:
                        metrics = extract_key_metrics(run_results)
                        
                        for metric_name, metric_value in metrics.items():
                            if metric_name not in metric_aggregates:
                                metric_aggregates[metric_name] = []
                            if metric_value is not None:
                                metric_aggregates[metric_name].append(metric_value)
            
            # Print statistics for key metrics
            key_metrics = ['accuracy', 'f1', 'precision', 'recall', 'mean_annotator_f1']
            
            for metric_name in key_metrics:
                if metric_name in metric_aggregates:
                    stats_result = calculate_statistics(metric_aggregates[metric_name])
                    print(f"{metric_name.upper()}:")
                    print(f"  Mean: {stats_result['mean']:.4f}")
                    print(f"  Std:  {stats_result['std']:.4f}")
                    print(f"  Min:  {stats_result['min']:.4f}")
                    print(f"  Max:  {stats_result['max']:.4f}")
                    print(f"  Count: {stats_result['count']}")
                    if stats_result['ci_95_lower'] is not None:
                        print(f"  95% CI: [{stats_result['ci_95_lower']:.4f}, {stats_result['ci_95_upper']:.4f}]")
                    print()
    
    print("=== End of Summary ===")

if __name__ == "__main__":
    main()
