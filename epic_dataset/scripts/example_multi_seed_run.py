#!/usr/bin/env python3
"""
Example script showing how to use the multi-seed experiment runner.

This script demonstrates various ways to run experiments with multiple seeds
and different configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_experiment_command(cmd_args, description):
    """Run an experiment command and print the description"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: python run_experiments_multi_seed.py {' '.join(cmd_args)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, "run_experiments_multi_seed.py"
        ] + cmd_args, 
        capture_output=True, text=True, check=True)
        
        print("✅ Experiment completed successfully!")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        if result.stderr:
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed with return code {e.returncode}")
        print("STDOUT:", e.stdout[-500:] if e.stdout else "None")
        print("STDERR:", e.stderr[-500:] if e.stderr else "None")

def main():
    """Run example experiments"""
    
    # Change to the scripts directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    print("Multi-Seed Experiment Runner Examples")
    print("=====================================")
    
    # Example 1: Basic multi-seed run with 3 approaches and 3 seeds
    print("\n1. Basic multi-seed experiment")
    cmd_args = [
        "--approaches", "aart", "multitask", "annotator_embedding",
        "--seeds", "42", "123", "456",
        "--runs_per_seed", "2"
    ]
    run_experiment_command(cmd_args, "Basic multi-seed run: 3 approaches × 3 seeds × 2 runs = 18 experiments")
    
    # Example 2: Single approach with more seeds and runs
    print("\n2. Single approach with many seeds")
    cmd_args = [
        "--approaches", "aart_rince",
        "--seeds", "1", "2", "3", "4", "5",
        "--runs_per_seed", "3"
    ]
    run_experiment_command(cmd_args, "Single approach with 5 seeds × 3 runs = 15 experiments")
    
    # Example 3: With noise and grouping
    print("\n3. With noise and grouping")
    cmd_args = [
        "--approaches", "aart", "annotator_embedding",
        "--seeds", "100", "200",
        "--runs_per_seed", "1",
        "--add_noise",
        "--noise_level", "0.3",
        "--use_grouping",
        "--annotators_per_group", "3"
    ]
    run_experiment_command(cmd_args, "With noise (0.3) and grouping (3 annotators per group)")
    
    # Example 4: With hyperparameter overrides
    print("\n4. With hyperparameter overrides")
    cmd_args = [
        "--approaches", "aart_rince",
        "--seeds", "999", "888",
        "--runs_per_seed", "1",
        "--lambda2", "0.2",
        "--temperature", "0.1",
        "--rince_lambda", "2.0"
    ]
    run_experiment_command(cmd_args, "With custom hyperparameters")
    
    # Example 5: Renegade noise strategy
    print("\n5. With renegade noise strategy")
    cmd_args = [
        "--approaches", "multitask", "annotator_embedding",
        "--seeds", "777", "666",
        "--runs_per_seed", "1",
        "--add_noise",
        "--noise_strategy", "renegade",
        "--renegade_percent", "0.2",
        "--renegade_flip_prob", "0.8"
    ]
    run_experiment_command(cmd_args, "With renegade noise (20% renegades, 80% flip probability)")
    
    print("\n" + "="*60)
    print("All example experiments completed!")
    print("Check the 'experiments/' directory for results.")
    print("Results will be in CSV format for easy analysis with pandas.")
    print("="*60)

if __name__ == "__main__":
    main()
