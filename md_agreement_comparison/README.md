# MD Agreement Analysis

This repository contains code for analyzing annotator agreement patterns in the TID-8 MD Agreement dataset using different modeling approaches.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Data

The dataset will be automatically downloaded from the Hugging Face Hub when running the download script:

```bash
python scripts/download_data.py
```

This will:
- Download the TID-8 dataset with "md-agreement-ann" configuration
- Process and save the data in `data/md_agreement/processed/`
- Create train and test splits in JSON format

## Running Experiments

### Basic Usage
Run all approaches without noise:

```bash
python scripts/run_experiments.py
```

### With Noise
Add annotator-specific noise:

```bash
python scripts/run_experiments.py --add_noise --noise_level 0.2
```

### Specific Approaches
Run only selected approaches:

```bash
python scripts/run_experiments.py --approaches aart multitask
```

Available approaches:
- `aart`: AART model for annotator agreement
- `multitask`: Multitask learning approach
- `annotator_embedding`: Annotator embedding model

Parameters:
- `--approaches`: List of approaches to run ['aart', 'multitask', 'annotator_embedding']
- `--add_noise`: Flag to enable noise injection
- `--noise_level`: Float between 0.0 and 1.0 (default: 0.2)

## Project Structure

```
md_agreement/
├── data/                    # Data directory (git-ignored)
│   └── md_agreement/
│       └── processed/      # Processed dataset files
├── models/
│   ├── checkpoints/        # Model checkpoints (git-ignored)
│   └── implementations/    # Model architecture implementations
├── scripts/                # Main scripts
│   ├── download_data.py   # Dataset download and processing
│   ├── run_experiments.py # Main experiment runner
│   └── ...
├── results/               # Experiment results (git-ignored)
└── logs/                 # Log files (git-ignored)
```

## Results

Results will be saved in:
- `results/final_comparison.txt`: Overall comparison of approaches
- `results/{approach}_detailed_metrics.json`: Detailed metrics per approach
- `logs/experiment.log`: Detailed experiment logs

Note: Data, checkpoints, and results directories are git-ignored.