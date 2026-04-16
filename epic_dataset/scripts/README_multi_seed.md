# Multi-Seed Experiment Runner

This directory contains a comprehensive multi-seed experiment runner for the MD Agreement project that allows you to run experiments with multiple random seeds and generate easy-to-analyze output.

## Files

- `run_experiments_multi_seed.py` - Main experiment runner script
- `analyze_results.py` - Analysis script for processing results
- `example_multi_seed_run.py` - Example usage script
- `README_multi_seed.md` - This documentation

## Features

### Multi-Seed Support
- Run experiments with multiple random seeds
- Specify number of runs per seed
- Automatic seed management for reproducibility

### Comprehensive Output
- **CSV format** for easy analysis with pandas
- **Statistical summaries** with mean, std, confidence intervals
- **Raw JSON results** for detailed inspection
- **Individual run results** for debugging

### Statistical Analysis Ready
- Pre-calculated statistics across seeds and runs
- 95% confidence intervals
- Easy integration with pandas for further analysis

## Usage

### Basic Usage

```bash
# Run 3 approaches with 3 seeds, 2 runs per seed (18 total experiments)
python run_experiments_multi_seed.py \
    --approaches aart multitask annotator_embedding \
    --seeds 42 123 456 \
    --runs_per_seed 2
```

### Advanced Usage

```bash
# With noise and grouping
python run_experiments_multi_seed.py \
    --approaches aart annotator_embedding \
    --seeds 100 200 300 \
    --runs_per_seed 3 \
    --add_noise \
    --noise_level 0.3 \
    --use_grouping \
    --annotators_per_group 4
```

### With Hyperparameter Overrides

```bash
# Custom hyperparameters
python run_experiments_multi_seed.py \
    --approaches aart_rince \
    --seeds 1 2 3 4 5 \
    --runs_per_seed 1 \
    --lambda2 0.2 \
    --temperature 0.1 \
    --rince_lambda 2.0
```

## Command Line Arguments

### Required Arguments
- `--approaches`: List of approaches to run (choices: majority_vote, aart, aart_rince, multitask, annotator_embedding, annotator_embedding_rince)
- `--seeds`: List of random seeds to use

### Optional Arguments
- `--runs_per_seed`: Number of runs per seed (default: 1)
- `--add_noise`: Add noise to labels during training
- `--noise_level`: Level of noise (default: 0.2)
- `--noise_strategy`: Noise strategy (fixed, random, custom, renegade)
- `--renegade_percent`: Percentage of renegade annotators (default: 0.1)
- `--renegade_flip_prob`: Probability of flipping labels for renegades (default: 0.7)
- `--use_grouping`: Enable annotator grouping
- `--annotators_per_group`: Number of annotators per group (default: 4)
- `--use_weighted_embeddings`: Use weighted embeddings
- `--experiment_id`: Custom experiment ID
- `--output_dir`: Base directory for outputs (default: experiments)

### Hyperparameter Overrides
- `--lambda2`: Override lambda2 hyperparameter
- `--contrastive_alpha`: Override contrastive_alpha hyperparameter
- `--temperature`: Override temperature hyperparameter
- `--rince_lambda`: Override rince_lambda hyperparameter
- `--rince_q`: Override rince_q hyperparameter

## Output Structure

Each experiment creates a directory structure like:

```
experiments/
└── {experiment_id}/
    ├── config.json                 # Experiment configuration
    ├── results/
    │   ├── results.csv            # Individual run results
    │   ├── statistical_summary.csv # Aggregated statistics
    │   └── raw_results.json       # Raw results
    └── logs/
        └── experiment.log         # Detailed logs
```

## CSV Output Format

### results.csv
Contains individual run results with columns:
- `experiment_id`, `approach`, `seed`, `run_number`
- Configuration parameters (noise, grouping, etc.)
- Performance metrics (accuracy, f1, precision, recall, etc.)
- Annotator-specific metrics

### statistical_summary.csv
Contains aggregated statistics with columns:
- `experiment_id`, `approach`, `metric`
- Statistics: `mean`, `std`, `min`, `max`, `count`
- Confidence intervals: `ci_95_lower`, `ci_95_upper`
- Configuration parameters

## Analysis

### Using the Analysis Script

```bash
# Basic analysis
python analyze_results.py --results_csv experiments/{experiment_id}/results/results.csv

# With plots
python analyze_results.py \
    --results_csv experiments/{experiment_id}/results/results.csv \
    --create_plots \
    --output_dir analysis_output
```

### Using Pandas Directly

```python
import pandas as pd

# Load results
df = pd.read_csv('experiments/{experiment_id}/results/results.csv')

# Basic statistics
print(df.groupby('approach')['accuracy'].agg(['mean', 'std', 'count']))

# Statistical tests
from scipy import stats
aart_acc = df[df['approach'] == 'aart']['accuracy']
multitask_acc = df[df['approach'] == 'multitask']['accuracy']
t_stat, p_value = stats.ttest_ind(aart_acc, multitask_acc)
print(f"t-test: t={t_stat:.4f}, p={p_value:.4f}")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='approach', y='accuracy')
plt.title('Accuracy by Approach')
plt.show()
```

## Examples

### Example 1: Quick Comparison
```bash
python run_experiments_multi_seed.py \
    --approaches aart multitask annotator_embedding \
    --seeds 42 123 456 \
    --runs_per_seed 1
```

### Example 2: Robust Evaluation
```bash
python run_experiments_multi_seed.py \
    --approaches aart_rince \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --runs_per_seed 3
```

### Example 3: Noise Robustness Study
```bash
python run_experiments_multi_seed.py \
    --approaches aart multitask annotator_embedding \
    --seeds 100 200 300 \
    --runs_per_seed 2 \
    --add_noise \
    --noise_level 0.3 \
    --noise_strategy renegade \
    --renegade_percent 0.2
```

## Tips

1. **Start Small**: Begin with 1-2 approaches and 2-3 seeds to test your setup
2. **Use Meaningful Seeds**: Use different seed ranges for different experiment types
3. **Monitor Resources**: More seeds/runs = longer execution time
4. **Check Logs**: Always check the experiment logs for errors
5. **Backup Results**: The CSV files are perfect for version control and sharing

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or number of parallel experiments
2. **CUDA Errors**: Check GPU availability and memory
3. **File Not Found**: Ensure data paths are correct in config
4. **Import Errors**: Check that all model implementations are available

### Debug Mode
Add `--experiment_id debug_test` to create a test experiment that's easy to identify and clean up.

## Performance Considerations

- **Total Experiments**: `len(approaches) × len(seeds) × runs_per_seed`
- **Execution Time**: Each experiment takes 10-30 minutes depending on approach
- **Storage**: Each experiment uses ~100MB for models and results
- **Memory**: Peak memory usage depends on batch size and model size

## Integration with Existing Code

The multi-seed runner is fully compatible with the existing codebase:
- Uses the same `ExperimentConfig` class
- Uses the same `Trainer` and model implementations
- Uses the same data loading pipeline
- Maintains the same evaluation metrics

The only difference is the orchestration layer that manages multiple seeds and runs.
