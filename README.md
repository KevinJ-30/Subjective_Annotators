# Experiments for Subjective Tasks

This repository contains two projects for comparing different approaches to handling subjective tasks:

1. Multi Domain Agreement (MD Agreement)
2. Sentiment Analysis

## Project Structure

```
.
├── md_agreement_comparison/
│   ├── data/                  # Data directory (gitignored)
│   ├── models/
│   │   └── implementations/   # Model implementations
│   └── scripts/              # Training and evaluation scripts
└── sentiment_analysis_comparison/
    ├── data/                  # Data directory (gitignored)
    ├── models/
    │   └── implementations/   # Model implementations
    └── scripts/              # Training and evaluation scripts
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
# For MD Agreement project
pip install -r md_agreement_comparison/requirements.txt

# For Sentiment Analysis project
pip install -r sentiment_analysis_comparison/requirements.txt
```

## Running Experiments

### Multi Domain Agreement

1. Download the dataset:
```bash
python md_agreement_comparison/scripts/download_data.py
```

2. Run experiments with different approaches:
```bash
# Basic run with all approaches
python md_agreement_comparison/scripts/run_experiments.py --approaches majority_vote aart multitask annotator_embedding

# With noise
python md_agreement_comparison/scripts/run_experiments.py --approaches majority_vote aart multitask annotator_embedding --add_noise --noise_level 0.2

# With annotator grouping
python md_agreement_comparison/scripts/run_experiments.py --approaches majority_vote aart multitask annotator_embedding --use_grouping --annotators_per_group 4

# With weighted embeddings (for annotator_embedding model)
python md_agreement_comparison/scripts/run_experiments.py --approaches majority_vote aart multitask annotator_embedding --use_weighted_embeddings
```

### Sentiment Analysis

1. Download the dataset:
```bash
python sentiment_analysis_comparison/scripts/download_data.py
```

2. Run experiments with different approaches:
```bash
# Basic run with all approaches
python sentiment_analysis_comparison/scripts/run_experiments.py --approaches majority_vote aart multitask annotator_embedding

# With noise
python sentiment_analysis_comparison/scripts/run_experiments.py --approaches majority_vote aart multitask annotator_embedding --add_noise --noise_level 0.2

# With annotator grouping
python sentiment_analysis_comparison/scripts/run_experiments.py --approaches majority_vote aart multitask annotator_embedding --use_grouping --annotators_per_group 4

# With weighted embeddings (for annotator_embedding model)
python sentiment_analysis_comparison/scripts/run_experiments.py --approaches majority_vote aart multitask annotator_embedding --use_weighted_embeddings
```

## Available Approaches

Both projects support the following approaches:

1. **Majority Vote**: Simple majority voting over annotator labels
2. **AART**: Agreement-Aware Representation Training
3. **Multitask**: Multitask learning approach
4. **Annotator Embedding**: Learning annotator embeddings

## Command Line Arguments

- `--approaches`: List of approaches to run (required)
- `--add_noise`: Add noise to labels during training
- `--noise_level`: Level of noise to add (default: 0.2)
- `--use_grouping`: Enable annotator grouping
- `--annotators_per_group`: Number of annotators per group (default: 4)
- `--use_weighted_embeddings`: Use weighted embeddings for annotator embedding model

## Output

Results are saved in the following locations:
- `results/final_comparison.txt`: Detailed comparison of all approaches
- `results/raw_results.json`: Raw results in JSON format
- `logs/experiment.log`: Detailed experiment logs
- `models/checkpoints/`: Model checkpoints for each approach

## Data Structure

### Multi Domain Agreement
- Binary classification task
- Each example has multiple annotator labels
- Labels are binary (0 or 1)

### Sentiment Analysis
- 5-class classification task
- Each example has multiple annotator labels
- Labels range from 0 to 4 (Very negative to Very positive)

## Notes

- Data directories are gitignored to prevent large files from being tracked
- Model checkpoints are saved during training
- Logs are generated for debugging and analysis
- Both projects use the same model architectures but are adapted for their specific tasks
