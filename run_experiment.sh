#!/bin/bash
###############################################################################
# run_experiment.sh — Launch a SINGLE dataset experiment inside a screen session
#
# This script runs one experiment for one dataset at a time.
# Edit the configuration variables below, then launch.
#
# For running MULTIPLE datasets in parallel, see run_all_experiments.sh
#
# ─── How to Run ─────────────────────────────────────────────────────────────
#
# 1. Start in DETACHED mode (runs in background, you get your terminal back):
#
#      screen -dmS my_exp bash run_experiment.sh
#
# 2. Start in INTERACTIVE mode (you see output immediately):
#
#      screen -S my_exp bash run_experiment.sh
#
# ─── How to Monitor ─────────────────────────────────────────────────────────
#
# List all running screen sessions:
#
#      screen -ls
#
# Reattach to a detached session to see live output:
#
#      screen -r my_exp
#
# If "screen -r" says "attached elsewhere", force reattach:
#
#      screen -d -r my_exp
#
# ─── How to Detach / Exit ───────────────────────────────────────────────────
#
# Detach from a session (leave it running in background):
#      Press Ctrl+A, then D
#
# Kill a session (stop the experiment):
#      screen -X -S my_exp quit
#
# ─── Tips ───────────────────────────────────────────────────────────────────
#
# - Give each experiment a unique screen name so you can track them:
#      screen -dmS md_aart bash run_experiment.sh
#      screen -dmS sent_rince bash run_experiment.sh
#
# - Scroll up inside a screen session:
#      Press Ctrl+A, then Esc, then use arrow keys / Page Up
#      Press Esc again to exit scroll mode
#
###############################################################################


# =============================================================================
# USER CONFIGURATION — Edit these variables before running
# =============================================================================

# ─── GPU Configuration ──────────────────────────────────────────────────────
# Which physical GPU to use. This sets CUDA_VISIBLE_DEVICES so that only this
# GPU is visible to PyTorch. The code internally always uses "cuda:0" because
# CUDA_VISIBLE_DEVICES remaps the selected GPU to index 0.
# Example: GPU_ID=2 means your physical GPU #2 becomes cuda:0 inside Python.
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ─── Dataset / Project ──────────────────────────────────────────────────────
# Which sub-project to run. Each maps to a folder in the repo:
#   "md_agreement" -> md_agreement_comparison/    (binary, 2 classes)
#   "sentiment"    -> sentiment_analysis_comparison/ (5 classes)
#   "hsb"          -> HSB-binary/                 (binary, 2 classes)
#   "epic"         -> epic_dataset/               (EPIC dataset)
DATASET="md_agreement"

# ─── Experiment Mode ────────────────────────────────────────────────────────
# "single"     -> runs run_experiments.py (one seed, hardcoded to 42)
# "multi_seed" -> runs run_experiments_multi_seed.py (multiple seeds, with
#                 statistical aggregation: mean, std, 95% CI across seeds)
MODE="multi_seed"

# ─── Approaches ─────────────────────────────────────────────────────────────
# Space-separated list of model approaches to train and evaluate.
#
# Available for MD / HSB:
#   majority_vote          - Simple majority voting (no training)
#   aart                   - Agreement-Aware Representation Training
#   aart_rince             - AART with RINCE contrastive loss
#   multitask              - One classification head per annotator
#   annotator_embedding    - Shared classifier with annotator embeddings
#   annotator_embedding_rince - Annotator embedding + RINCE loss
#
# Additional for Sentiment:
#   aart_ord_rince         - AART with ordinal RINCE loss
#   aart_likert            - AART with Likert ranking-robust loss
APPROACHES="aart multitask annotator_embedding"

# ─── Training Configuration ────────────────────────────────────────────────
# Number of training epochs. Leave empty to use the default from each
# dataset's config.py (MD=1, Sentiment=10, HSB=10).
NUM_EPOCHS=""

# ─── Seeds (multi_seed mode only) ───────────────────────────────────────────
# Space-separated list of random seeds. Each seed produces an independent
# training run, and results are aggregated across all seeds.
SEEDS="42 123 456"

# How many times to repeat each seed. Useful for measuring variance even
# within the same seed (e.g., due to non-deterministic GPU operations).
RUNS_PER_SEED=1

# ─── Noise Configuration ────────────────────────────────────────────────────
# Simulate noisy annotators by flipping labels during training.
#
# Noise strategy (used for all noisy runs):
#   "fixed"    - All annotators get the same flip probability
#   "random"   - Each annotator gets a random flip prob between 0.1-0.3
#   "custom"   - Use a custom noise_levels dict (set in code)
#   "renegade" - A small % of annotators get high noise, rest get 0%
NOISE_STRATEGY="fixed"

# Noise levels to sweep over (space-separated).
# The script runs ALL approaches for each noise level, one after another.
#
# Use "none" to run WITHOUT noise. You can mix "none" with numeric levels
# to run a clean baseline followed by noisy experiments in one launch.
#
# Examples:
#   NOISE_LEVELS="none"                         # No noise at all
#   NOISE_LEVELS="0.2"                          # Single noise level
#   NOISE_LEVELS="none 0.1 0.2 0.3"            # Clean run first, then 3 noise levels
#   NOISE_LEVELS="none 0.0 0.1 0.2 0.3 0.4"   # Full sweep including baseline
NOISE_LEVELS="none"

# Renegade-specific settings (only used when NOISE_STRATEGY="renegade"):
RENEGADE_PERCENT=0.1             # 10% of annotators become renegades
RENEGADE_FLIP_PROB=0.7           # Renegades flip labels 70% of the time

# ─── Annotator Grouping ─────────────────────────────────────────────────────
# Cluster annotators into groups to simulate fewer, more reliable annotators.
# Uses Cohen's kappa + hierarchical clustering (MD/HSB) or KMeans (sentiment).
USE_GROUPING=false                # Set to true to enable grouping
ANNOTATORS_PER_GROUP=4           # Target number of annotators per group

# ─── Model Options ──────────────────────────────────────────────────────────
# Use attention-weighted annotator embeddings (for annotator_embedding model).
# When true, a small attention layer learns to weight embeddings.
USE_WEIGHTED_EMBEDDINGS=false

# ─── Hyperparameter Overrides ───────────────────────────────────────────────
# Leave empty ("") to use the defaults defined in each approach's setup.
# Only set these if you want to override for a specific experiment.
LAMBDA2=""                       # Contrastive loss weight (AART/RINCE)
CONTRASTIVE_ALPHA=""             # Contrastive alpha (AART)
TEMPERATURE=""                   # Temperature for contrastive loss (RINCE)
RINCE_LAMBDA=""                  # RINCE lambda parameter

# RINCE q parameter — controls robustness to noise in RINCE-based approaches.
# Only has effect when noise is present (ignored for "none" noise levels).
# Space-separated list to sweep over multiple values per noise level.
# Leave empty for approach defaults.
# Examples:
#   RINCE_Q_VALUES=""                   # Use approach default
#   RINCE_Q_VALUES="0.5"               # Single value
#   RINCE_Q_VALUES="0.1 0.5 1.0"       # Sweep 3 values per noise level
RINCE_Q_VALUES=""

# ─── Virtual Environment ────────────────────────────────────────────────────
# Path to the Python virtual environment, relative to the project root.
VENV_PATH="venv"


# =============================================================================
# EXECUTION LOGIC — You should not need to edit below this line
# =============================================================================

# Exit on error, undefined variables, or pipe failures
set -euo pipefail

# cd to the project root (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the Python virtual environment
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated virtualenv: $VENV_PATH"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Map the DATASET name to the actual folder and pick the right Python script
case "$DATASET" in
    md_agreement)
        PROJECT_DIR="md_agreement_comparison"
        ;;
    sentiment)
        PROJECT_DIR="sentiment_analysis_comparison"
        ;;
    hsb)
        PROJECT_DIR="HSB-binary"
        ;;
    epic)
        PROJECT_DIR="epic_dataset"
        ;;
    *)
        echo "ERROR: Unknown DATASET '$DATASET'. Use: md_agreement | sentiment | hsb | epic"
        exit 1
        ;;
esac

# Choose between single-seed and multi-seed experiment script
if [ "$MODE" = "multi_seed" ]; then
    SCRIPT_NAME="run_experiments_multi_seed.py"
else
    SCRIPT_NAME="run_experiments.py"
fi

SCRIPT_PATH="$PROJECT_DIR/scripts/$SCRIPT_NAME"

# Sanity check: make sure the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Helper function to build and run one experiment.
# Args: $1 = noise_level ("none" or a number), $2 = rince_q (optional, "" for default)
run_single_config() {
    local noise_level=$1
    local rince_q=${2:-""}
    local is_noisy=true

    if [ "$noise_level" = "none" ]; then
        is_noisy=false
    fi

    # Start building the Python command
    local CMD="python $SCRIPT_PATH --approaches $APPROACHES"

    # Append seed arguments (only relevant for multi_seed mode)
    if [ "$MODE" = "multi_seed" ]; then
        CMD="$CMD --seeds $SEEDS --runs_per_seed $RUNS_PER_SEED"
    fi

    # Append noise flags only for noisy runs (not for "none")
    if [ "$is_noisy" = true ]; then
        CMD="$CMD --add_noise --noise_strategy $NOISE_STRATEGY --noise_level $noise_level"
        if [ "$NOISE_STRATEGY" = "renegade" ]; then
            CMD="$CMD --renegade_percent $RENEGADE_PERCENT --renegade_flip_prob $RENEGADE_FLIP_PROB"
        fi
    fi

    # Append grouping flags if annotator grouping is enabled
    if [ "$USE_GROUPING" = true ]; then
        CMD="$CMD --use_grouping --annotators_per_group $ANNOTATORS_PER_GROUP"
    fi

    # Append weighted embeddings flag
    if [ "$USE_WEIGHTED_EMBEDDINGS" = true ]; then
        CMD="$CMD --use_weighted_embeddings"
    fi

    # Append hyperparameter overrides only if they are set (non-empty)
    [ -n "$LAMBDA2" ]            && CMD="$CMD --lambda2 $LAMBDA2"
    [ -n "$CONTRASTIVE_ALPHA" ]  && CMD="$CMD --contrastive_alpha $CONTRASTIVE_ALPHA"
    [ -n "$TEMPERATURE" ]        && CMD="$CMD --temperature $TEMPERATURE"
    [ -n "$RINCE_LAMBDA" ]       && CMD="$CMD --rince_lambda $RINCE_LAMBDA"
    [ -n "$rince_q" ]            && CMD="$CMD --rince_q $rince_q"
    [ -n "$NUM_EPOCHS" ]         && CMD="$CMD --num_epochs $NUM_EPOCHS"

    # Format display strings
    local noise_display="OFF"
    if [ "$is_noisy" = true ]; then
        noise_display="ON (strategy=$NOISE_STRATEGY, level=$noise_level)"
    fi

    local rince_q_display="default"
    [ -n "$rince_q" ] && rince_q_display="$rince_q"

    # Print config for this run
    echo "============================================================"
    echo "  Experiment Configuration"
    echo "============================================================"
    echo "  GPU:              $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
    echo "  Dataset:          $DATASET ($PROJECT_DIR)"
    echo "  Mode:             $MODE"
    echo "  Approaches:       $APPROACHES"
    if [ "$MODE" = "multi_seed" ]; then
    echo "  Seeds:            $SEEDS"
    echo "  Runs per seed:    $RUNS_PER_SEED"
    fi
    echo "  Noise:            $noise_display"
    if [ "$is_noisy" = true ]; then
    echo "  RINCE q:          $rince_q_display"
    fi
    echo "  Grouping:         $USE_GROUPING (per_group=$ANNOTATORS_PER_GROUP)"
    echo "  Weighted embeds:  $USE_WEIGHTED_EMBEDDINGS"
    echo "------------------------------------------------------------"
    echo "  Command:"
    echo "    $CMD"
    echo "============================================================"
    echo ""

    $CMD
}

# Build the list of (noise_level, rince_q) configurations to run
configs=()
for noise_level in $NOISE_LEVELS; do
    if [ "$noise_level" = "none" ]; then
        # Clean run: no rince_q sweep
        configs+=("$noise_level|")
    elif [ -n "$RINCE_Q_VALUES" ]; then
        # Noisy run with rince_q sweep
        for rq in $RINCE_Q_VALUES; do
            configs+=("$noise_level|$rq")
        done
    else
        # Noisy run without rince_q sweep
        configs+=("$noise_level|")
    fi
done

total_configs=${#configs[@]}

if [ "$total_configs" -gt 1 ]; then
    # ── SWEEP MODE ───────────────────────────────────────────────────────────
    echo ""
    echo "============================================================"
    echo "  SWEEP: $total_configs configurations"
    echo "  Noise levels: $NOISE_LEVELS"
    if [ -n "$RINCE_Q_VALUES" ]; then
    echo "  RINCE q values: $RINCE_Q_VALUES (noisy runs only)"
    fi
    echo "============================================================"

    current=0
    for config_pair in "${configs[@]}"; do
        current=$((current + 1))
        noise_level="${config_pair%%|*}"
        rince_q="${config_pair##*|}"

        label="noise=$noise_level"
        [ "$noise_level" = "none" ] && label="no noise (clean baseline)"
        [ -n "$rince_q" ] && label="$label, rince_q=$rince_q"

        echo ""
        echo "************************************************************"
        echo "  Run $current/$total_configs: $label"
        echo "************************************************************"
        run_single_config "$noise_level" "$rince_q"
    done

    echo ""
    echo "============================================================"
    echo "  Sweep complete: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
else
    # ── SINGLE RUN MODE ──────────────────────────────────────────────────────
    noise_level="${configs[0]%%|*}"
    rince_q="${configs[0]##*|}"
    run_single_config "$noise_level" "$rince_q"
fi
