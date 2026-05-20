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
import uuid
import random

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Initialize logging (may be reconfigured in main() for per-experiment logs)
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

# Import models
from models.implementations.multitask import MultitaskModel
from models.implementations.aart import AARTModel
from models.implementations.aart_Rince_new import NewRinceModel
from models.implementations.annotator_embedding import AnnotatorEmbeddingModel
from models.implementations.majority_vote import MajorityVoteModel
# from models.implementations.annotator_embedding_rince import AnnotatorEmbeddingRinceModel


def setup_config(
    approach,
    add_noise=False,
    noise_level=0.2,
    noise_strategy='fixed',
    renegade_percent=0.1,
    renegade_flip_prob=0.7,
    use_grouping=False,
    annotators_per_group=4
):
    """Setup configuration for a specific approach (base defaults + approach defaults)."""
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
        annotators_per_group=annotators_per_group
    )

    # Set approach-specific parameters (defaults; may be overridden later)
    if approach == 'majority_vote':
        config.use_majority_vote = True

    elif approach == 'aart':
        config.lambda2 = 0.1
        config.contrastive_alpha = 0.1

    elif approach == 'aart_rince':
        config.lambda2 = 0.1
        config.contrastive_alpha = 0.1
        config.temperature = 0.07
        config.rince_lambda = 0.5
        config.rince_q = 0.5

    elif approach == 'multitask':
        pass

    elif approach == 'annotator_embedding':
        config.use_annotator_embed = True
        config.use_annotation_embed = True

    elif approach == 'annotator_embedding_rince':
        config.use_annotator_embed = True
        config.use_annotation_embed = True
        config.lambda2 = 0.1
        config.temperature = 0.07
        config.rince_lambda = 1.0
        config.rince_q = 1.0

    return config


def apply_overrides(config: ExperimentConfig, overrides: dict):
    """
    Apply CLI overrides onto ExperimentConfig (dataclass instance).
    Uses setattr instead of dict.update.
    """
    for k, v in overrides.items():
        if v is None:
            continue
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            # Some configs are plain dataclasses without slots; setting is usually ok.
            # If ExperimentConfig uses slots, this will raise AttributeError (which is useful).
            logging.warning(f"[config] ExperimentConfig has no attribute '{k}'. Setting it anyway.")
            setattr(config, k, v)


def run_single_experiment(
    approach,
    experiment_id,
    add_noise=False,
    noise_level=0.2,
    noise_strategy='fixed',
    renegade_percent=0.1,
    renegade_flip_prob=0.7,
    use_grouping=False,
    annotators_per_group=4,
    use_weighted_embeddings=False,
    # ---- overrides (optional) ----
    seed=None,
    num_epochs=None,
    lambda2=None,
    temperature=None,
    rince_q=None,
    rince_lambda=None,
):
    """Run a single experiment with the specified approach."""
    try:
        if approach != 'aart_rince':  # Only log for non-aart_rince approaches
            logging.info(f"\nStarting experiment for {approach}")

        # Create config
        config = setup_config(
            approach=approach,
            add_noise=add_noise,
            noise_level=noise_level,
            noise_strategy=noise_strategy,
            renegade_percent=renegade_percent,
            renegade_flip_prob=renegade_flip_prob,
            use_grouping=use_grouping,
            annotators_per_group=annotators_per_group
        )

        # Apply CLI overrides
        apply_overrides(config, {
            "seed": seed,
            "num_epochs": num_epochs,
            "lambda2": lambda2,
            "temperature": temperature,
            "rince_q": rince_q,
            "rince_lambda": rince_lambda,
        })

        # Weighted embeddings if requested
        if use_weighted_embeddings:
            config.use_weighted_embeddings = True

        # Ensure seed is set in config
        if not hasattr(config, 'seed') or config.seed is None:
            config.seed = 150

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
            trainer = Trainer(config, NewRinceModel)
        elif approach == 'multitask':
            trainer = Trainer(config, MultitaskModel)
        elif approach == 'annotator_embedding':
            trainer = Trainer(config, AnnotatorEmbeddingModel)
        elif approach == 'annotator_embedding_rince':
            trainer = Trainer(config, AnnotatorEmbeddingRinceModel)
        else:
            raise ValueError(f"Unknown approach: {approach}")

        # Train and evaluate (trainer.train() already evaluates and returns metrics)
        results = trainer.train()

        if approach != 'aart_rince':
            logging.info(f"Training completed for {approach}")

        return results

    except Exception as e:
        logging.error(f"Error running {approach}: {str(e)}")
        traceback.print_exc()
        return None


def write_final_comparison(results, output_path):
    """Write comparison of results with detailed metrics."""
    logging.info("Writing final comparison...")

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except (TypeError, ValueError):
            return "N/A"

    with open(output_path, 'w') as f:
        f.write("=== Final Comparison of All Approaches ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        approaches = sorted(results.keys())

        for approach in approaches:
            f.write(f"\n=== {approach.upper()} ===\n")
            if approach in results and isinstance(results[approach], dict):
                metrics = results[approach]

                f.write("\nOverall Metrics:\n")
                f.write(f"Accuracy: {fmt(metrics.get('accuracy'))}\n")
                f.write(f"F1 Score: {fmt(metrics.get('f1'))}\n")
                f.write(f"Precision: {fmt(metrics.get('precision'))}\n")
                f.write(f"Recall: {fmt(metrics.get('recall'))}\n")

                # Per-class metrics (if present)
                f.write("\nPer-Class Metrics:\n")
                num_classes = int(metrics.get('num_classes', 2)) if str(metrics.get('num_classes', 2)).isdigit() else 2
                for i in range(num_classes):
                    f.write(f"Class {i}:\n")
                    f.write(f"  F1: {fmt(metrics.get(f'class_{i}_f1'))}\n")
                    f.write(f"  Precision: {fmt(metrics.get(f'class_{i}_precision'))}\n")
                    f.write(f"  Recall: {fmt(metrics.get(f'class_{i}_recall'))}\n")

                f.write("\nAnnotator Metrics:\n")
                f.write(f"Mean Annotator F1: {fmt(metrics.get('mean_annotator_f1'))}\n")
                f.write(f"Std Annotator F1: {fmt(metrics.get('std_annotator_f1'))}\n")
                f.write(f"Min Annotator F1: {fmt(metrics.get('min_annotator_f1'))}\n")
                f.write(f"Max Annotator F1: {fmt(metrics.get('max_annotator_f1'))}\n")
                f.write(f"Number of Annotators Evaluated: {metrics.get('num_annotators_evaluated', 'N/A')}\n")

                if 'per_annotator_metrics' in metrics:
                    f.write("\nIndividual Annotator Metrics:\n")
                    for annotator_id, annotator_metrics in metrics['per_annotator_metrics'].items():
                        f.write(f"Annotator {annotator_id}:\n")
                        f.write(f"  F1: {fmt(annotator_metrics.get('f1'))}\n")
                        f.write(f"  Accuracy: {fmt(annotator_metrics.get('accuracy'))}\n")
                        f.write(f"  Samples: {annotator_metrics.get('num_samples', 'N/A')}\n")
            else:
                f.write("No results available\n")

        f.write("\n=== End of Report ===\n")


def set_seeds(seed=150):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_experiment_id(args):
    """Generate a unique experiment ID based on parameters."""
    unique_id = str(uuid.uuid4())[:8]

    name_parts = []
    approach_str = '_'.join(sorted(args.approaches))
    name_parts.append(f"approaches-{approach_str}")

    if args.use_grouping:
        name_parts.append(f"grouping-{args.annotators_per_group}")

    if args.add_noise:
        if args.noise_strategy == 'renegade':
            name_parts.append(f"renegade-{args.renegade_percent}-{args.renegade_flip_prob}")
        else:
            name_parts.append(f"noise-{args.noise_level}")

    if args.use_weighted_embeddings:
        name_parts.append("weighted")

    # record key overrides in experiment name (helps later)
    if args.seed is not None:
        name_parts.append(f"seed-{args.seed}")
    if args.num_epochs is not None:
        name_parts.append(f"ep-{args.num_epochs}")
    if args.lambda2 is not None:
        name_parts.append(f"lam2-{args.lambda2}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts.append(timestamp)

    experiment_name = "__".join(name_parts)
    experiment_id = f"{experiment_name}_{unique_id}"
    return experiment_id


def fmt4(x):
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def print_final_summary(results):
    """Print a summary of the results to the console (safe formatting)."""
    print("\n=== Final Results Summary ===")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for approach in sorted(results.keys()):
        print(f"\n=== {approach.upper()} ===")
        if approach in results and isinstance(results[approach], dict):
            metrics = results[approach]

            print("\nOverall Metrics:")
            print(f"Accuracy: {fmt4(metrics.get('accuracy'))}")
            print(f"F1 Score: {fmt4(metrics.get('f1'))}")
            print(f"Precision: {fmt4(metrics.get('precision'))}")
            print(f"Recall: {fmt4(metrics.get('recall'))}")

            print("\nPer-Class Metrics:")
            num_classes = metrics.get('num_classes', 2)
            try:
                num_classes = int(num_classes)
            except Exception:
                num_classes = 2
            for i in range(num_classes):
                print(f"Class {i}:")
                print(f"  F1: {fmt4(metrics.get(f'class_{i}_f1'))}")
                print(f"  Precision: {fmt4(metrics.get(f'class_{i}_precision'))}")
                print(f"  Recall: {fmt4(metrics.get(f'class_{i}_recall'))}")

            print("\nAnnotator Metrics:")
            print(f"Mean Annotator F1: {fmt4(metrics.get('mean_annotator_f1'))}")
            print(f"Std Annotator F1: {fmt4(metrics.get('std_annotator_f1'))}")
            print(f"Min Annotator F1: {fmt4(metrics.get('min_annotator_f1'))}")
            print(f"Max Annotator F1: {fmt4(metrics.get('max_annotator_f1'))}")
            print(f"Number of Annotators Evaluated: {metrics.get('num_annotators_evaluated', 'N/A')}")
        else:
            print("No results available")

    print("\n=== End of Summary ===")


def main():
    parser = argparse.ArgumentParser(description='Run MD agreement experiments')
    parser.add_argument(
        '--approaches',
        nargs='+',
        required=True,
        choices=[
            'majority_vote',
            'aart',
            'aart_rince',
            'multitask',
            'annotator_embedding',
            'annotator_embedding_rince'
        ],
        help='Approaches to run'
    )
    parser.add_argument('--use_weighted_embeddings', action='store_true',
                        help='Use weighted embeddings for annotator embedding model')
    parser.add_argument('--add_noise', action='store_true',
                        help='Add noise to labels during training')
    parser.add_argument('--noise_level', type=float, default=0.2,
                        help='Level of noise to add to labels (default: 0.2)')
    parser.add_argument('--noise_strategy', type=str, default='fixed',
                        choices=['fixed', 'random', 'custom', 'renegade'],
                        help='Strategy for adding noise (default: fixed)')
    parser.add_argument('--renegade_percent', type=float, default=0.1,
                        help='Percentage of annotators to be renegades (default: 0.1)')
    parser.add_argument('--renegade_flip_prob', type=float, default=0.7,
                        help='Probability of flipping labels for renegade annotators (default: 0.7)')
    parser.add_argument('--use_grouping', action='store_true',
                        help='Enable annotator grouping')
    parser.add_argument('--annotators_per_group', type=int, default=4,
                        help='Number of annotators per group when grouping is enabled')
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='Optional experiment ID to use (if not provided, a new one will be generated)')

    # ---- NEW override args ----
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed override (applied to all approaches)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Epoch override (applied to all approaches)')
    parser.add_argument('--lambda2', type=float, default=None,
                        help='lambda2 override (applied to approaches that use it)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='temperature override (RINCE models)')
    parser.add_argument('--rince_q', type=float, default=None,
                        help='RINCE q override')
    parser.add_argument('--rince_lambda', type=float, default=None,
                        help='RINCE lambda override')

    args = parser.parse_args()

    experiment_id = args.experiment_id if args.experiment_id else get_experiment_id(args)

    experiment_dir = Path("experiments") / experiment_id
    results_dir = experiment_dir / "results"
    logs_dir = experiment_dir / "logs"
    for dir_path in [results_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / 'experiment.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Starting experiment: {experiment_id}")
    logging.info(f"Selected approaches: {args.approaches}")
    logging.info(f"Results will be saved in: {experiment_dir}")

    # Save experiment configuration (includes overrides)
    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        cfg = {
            'experiment_id': experiment_id,
            'approaches': args.approaches,
            'use_weighted_embeddings': args.use_weighted_embeddings,
            'add_noise': args.add_noise,
            'noise_level': args.noise_level,
            'noise_strategy': args.noise_strategy,
            'renegade_percent': args.renegade_percent,
            'renegade_flip_prob': args.renegade_flip_prob,
            'use_grouping': args.use_grouping,
            'annotators_per_group': args.annotators_per_group,
            'seed': args.seed,
            'num_epochs': args.num_epochs,
            'lambda2': args.lambda2,
            'temperature': args.temperature,
            'rince_q': args.rince_q,
            'rince_lambda': args.rince_lambda,
            'timestamp': datetime.now().isoformat()
        }
        json.dump(cfg, f, indent=2)

    # Set random seeds for reproducibility (override if provided)
    set_seeds(args.seed if args.seed is not None else 150)

    results = {}
    for approach in args.approaches:
        try:
            print(f"\nDebug - Running {approach}")
            results[approach] = run_single_experiment(
                approach,
                experiment_id=experiment_id,
                add_noise=args.add_noise,
                noise_level=args.noise_level,
                noise_strategy=args.noise_strategy,
                renegade_percent=args.renegade_percent,
                renegade_flip_prob=args.renegade_flip_prob,
                use_grouping=args.use_grouping,
                annotators_per_group=args.annotators_per_group,
                use_weighted_embeddings=args.use_weighted_embeddings,
                # overrides:
                seed=args.seed,
                num_epochs=args.num_epochs,
                lambda2=args.lambda2,
                temperature=args.temperature,
                rince_q=args.rince_q,
                rince_lambda=args.rince_lambda,
            )
            print(f"\nDebug - Results for {approach}:")
            print(f"Results type: {type(results[approach])}")
            print(f"Results content: {results[approach]}")
        except Exception as e:
            print(f"\nError running {approach}:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            results[approach] = {"error": str(e)}

    try:
        comparison_path = results_dir / "final_comparison.txt"
        print(f"\nDebug - Writing comparison to {comparison_path}")
        write_final_comparison(results, comparison_path)

        raw_results_path = results_dir / "raw_results.json"
        print(f"\nDebug - Writing raw results to {raw_results_path}")
        print(f"Raw results content: {results}")
        with open(raw_results_path, 'w') as f:
            json.dump(results, f, indent=2)

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

        print_final_summary(results)

    except Exception as e:
        logging.error(f"Error writing results: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()