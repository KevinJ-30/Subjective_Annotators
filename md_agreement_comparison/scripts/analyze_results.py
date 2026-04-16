#!/usr/bin/env python3
"""
Analysis script for multi-seed experiment results.

This script demonstrates how to load and analyze the CSV results
from the multi-seed experiment runner.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_results(csv_path):
    """Load results from CSV file"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiment results from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df

def basic_summary(df):
    """Print basic summary of results"""
    print("\n" + "="*60)
    print("BASIC SUMMARY")
    print("="*60)
    
    print(f"Total experiments: {len(df)}")
    print(f"Approaches: {df['approach'].unique()}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Runs per seed: {df.groupby(['approach', 'seed']).size().iloc[0]}")
    
    # Check for missing values
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count}")

def performance_comparison(df):
    """Compare performance across approaches"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    key_metrics = ['accuracy', 'f1', 'precision', 'recall', 'mean_annotator_f1']
    
    for metric in key_metrics:
        if metric in df.columns:
            print(f"\n{metric.upper()}:")
            print("-" * 40)
            
            # Group by approach and calculate statistics
            stats_df = df.groupby('approach')[metric].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)
            
            print(stats_df)
            
            # Statistical significance test (if multiple approaches)
            approaches = df['approach'].unique()
            if len(approaches) > 1:
                print(f"\nStatistical significance tests for {metric}:")
                for i, approach1 in enumerate(approaches):
                    for approach2 in approaches[i+1:]:
                        data1 = df[df['approach'] == approach1][metric].dropna()
                        data2 = df[df['approach'] == approach2][metric].dropna()
                        
                        if len(data1) > 1 and len(data2) > 1:
                            t_stat, p_value = stats.ttest_ind(data1, data2)
                            print(f"  {approach1} vs {approach2}: t={t_stat:.4f}, p={p_value:.4f}")

def seed_analysis(df):
    """Analyze results across different seeds"""
    print("\n" + "="*60)
    print("SEED ANALYSIS")
    print("="*60)
    
    key_metrics = ['accuracy', 'f1', 'mean_annotator_f1']
    
    for metric in key_metrics:
        if metric in df.columns:
            print(f"\n{metric.upper()} by seed:")
            print("-" * 40)
            
            # Group by seed and approach
            seed_stats = df.groupby(['seed', 'approach'])[metric].agg([
                'mean', 'std', 'count'
            ]).round(4)
            
            print(seed_stats)
            
            # Check for seed effects
            print(f"\nSeed effect analysis for {metric}:")
            for approach in df['approach'].unique():
                approach_data = df[df['approach'] == approach]
                if len(approach_data) > 1:
                    # ANOVA test across seeds
                    seed_groups = [group[metric].dropna().values 
                                 for name, group in approach_data.groupby('seed')]
                    
                    if all(len(group) > 0 for group in seed_groups):
                        f_stat, p_value = stats.f_oneway(*seed_groups)
                        print(f"  {approach}: F={f_stat:.4f}, p={p_value:.4f}")

def noise_analysis(df):
    """Analyze the effect of noise"""
    print("\n" + "="*60)
    print("NOISE ANALYSIS")
    print("="*60)
    
    if 'add_noise' not in df.columns:
        print("No noise information available in results")
        return
    
    noise_experiments = df[df['add_noise'] == True]
    clean_experiments = df[df['add_noise'] == False]
    
    if len(noise_experiments) == 0:
        print("No noise experiments found")
        return
    
    if len(clean_experiments) == 0:
        print("No clean experiments found")
        return
    
    print(f"Noise experiments: {len(noise_experiments)}")
    print(f"Clean experiments: {len(clean_experiments)}")
    
    key_metrics = ['accuracy', 'f1', 'mean_annotator_f1']
    
    for metric in key_metrics:
        if metric in df.columns:
            print(f"\n{metric.upper()} - Noise vs Clean:")
            print("-" * 40)
            
            noise_values = noise_experiments[metric].dropna()
            clean_values = clean_experiments[metric].dropna()
            
            if len(noise_values) > 0 and len(clean_values) > 0:
                print(f"Noise:   mean={noise_values.mean():.4f}, std={noise_values.std():.4f}, n={len(noise_values)}")
                print(f"Clean:   mean={clean_values.mean():.4f}, std={clean_values.std():.4f}, n={len(clean_values)}")
                
                # Statistical test
                t_stat, p_value = stats.ttest_ind(noise_values, clean_values)
                print(f"t-test:  t={t_stat:.4f}, p={p_value:.4f}")

def create_plots(df, output_dir):
    """Create visualization plots"""
    print("\n" + "="*60)
    print("CREATING PLOTS")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    key_metrics = ['accuracy', 'f1', 'mean_annotator_f1']
    
    for metric in key_metrics:
        if metric in df.columns:
            # Box plot by approach
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='approach', y=metric)
            plt.title(f'{metric.upper()} by Approach')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_by_approach.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Violin plot by approach
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=df, x='approach', y=metric)
            plt.title(f'{metric.upper()} Distribution by Approach')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_distribution_by_approach.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Line plot by seed
            plt.figure(figsize=(12, 6))
            for approach in df['approach'].unique():
                approach_data = df[df['approach'] == approach]
                seed_means = approach_data.groupby('seed')[metric].mean()
                plt.plot(seed_means.index, seed_means.values, marker='o', label=approach)
            plt.xlabel('Seed')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} by Seed')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric}_by_seed.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze multi-seed experiment results')
    parser.add_argument('--results_csv', type=str, required=True,
                      help='Path to the results CSV file')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                      help='Directory to save analysis plots (default: analysis_output)')
    parser.add_argument('--create_plots', action='store_true',
                      help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.results_csv)
    
    # Run analyses
    basic_summary(df)
    performance_comparison(df)
    seed_analysis(df)
    noise_analysis(df)
    
    # Create plots if requested
    if args.create_plots:
        create_plots(df, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
