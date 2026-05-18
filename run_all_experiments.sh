#!/bin/bash
###############################################################################
# run_all_experiments.sh — Smart GPU scheduler for multi-dataset experiments
#
# Features:
#   - Per-approach job queuing: each (dataset, approach) pair is a separate job
#   - Memory-aware GPU packing: multiple jobs can share a GPU if memory allows
#   - Per-dataset memory requirements: different datasets can request different
#     amounts of GPU memory (e.g., sentiment needs more than MD)
#   - Auto-retry on OOM: if a job fails due to out-of-memory, it is
#     automatically re-queued with a higher memory requirement
#   - Three execution modes: parallel, sequential, queued
#
# ─── How to Run ─────────────────────────────────────────────────────────────
#
# 1. Start in DETACHED mode (runs in background, you get your terminal back):
#
#      screen -dmS experiments bash run_all_experiments.sh
#
# 2. Start in INTERACTIVE mode (you see output immediately):
#
#      screen -S experiments bash run_all_experiments.sh
#
# ─── How to Monitor ─────────────────────────────────────────────────────────
#
# List all running screen sessions:
#
#      screen -ls
#
# Reattach to a detached session to see live output:
#
#      screen -r experiments
#
# If "screen -r" says "attached elsewhere", force reattach:
#
#      screen -d -r experiments
#
# Check GPU usage while experiments are running (from another terminal):
#
#      watch -n 2 nvidia-smi
#
# Check job logs (each job saves output to logs/gpu_scheduler/):
#
#      ls -lt logs/gpu_scheduler/
#      tail -f logs/gpu_scheduler/<logfile>.log
#
# ─── How to Detach / Exit ───────────────────────────────────────────────────
#
# Detach from a session (leave it running in background):
#      Press Ctrl+A, then D
#
# Kill a session (stop ALL experiments):
#      screen -X -S experiments quit
#
# ─── Tips ───────────────────────────────────────────────────────────────────
#
# - Scroll up inside a screen session:
#      Press Ctrl+A, then Esc, then use arrow keys / Page Up
#      Press Esc again to exit scroll mode
#
# - After experiments finish, results are in multi_seed_experiments/ or
#   experiments/ depending on the MODE setting.
#
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# ─── Execution Mode ─────────────────────────────────────────────────────────
# "parallel"   - Each dataset gets a DEDICATED GPU (set below via MD_GPU,
#                SENTIMENT_GPU, HSB_GPU). All approaches, noise levels, and
#                seeds for that dataset run sequentially on its GPU. No memory
#                contention — only one job per GPU at a time. Best when you
#                have enough free GPUs and want reliable, no-OOM runs.
#                GPU_POOL is NOT used in this mode.
#
# "sequential" - Everything runs on ONE GPU (SEQ_GPU), one job at a time.
#                Safest option but slowest. GPU_POOL is NOT used.
#
# "queued"     - Each (dataset, approach, noise_level) triple is an independent
#                job. Jobs auto-claim GPUs from GPU_POOL based on free memory.
#                Multiple jobs CAN share a GPU if memory allows (memory-aware
#                packing). Fastest when GPUs have headroom, but risks OOM if
#                the server is busy. Uses GPU_POOL, per-dataset memory
#                requirements, and retry settings below.
EXEC_MODE="parallel"

# ─── GPU Configuration (queued mode only) ──────────────────────────────────
# Comma-separated list of physical GPU IDs the scheduler can claim from.
# ONLY used by "queued" mode. Ignored by "parallel" and "sequential".
GPU_POOL="3,4,5"

# How often (in seconds) to poll for a free GPU when all are busy.
# ONLY used by "queued" mode.
QUEUE_POLL_INTERVAL=10

# ─── Per-Dataset Memory Requirements (queued mode only, in MB) ──────────────
# The scheduler will only place a job on a GPU that has at least this much
# free memory. ONLY used by "queued" mode — "parallel" and "sequential"
# assume exclusive use of their assigned GPU.
#
# Tips for setting these values:
#   - Run one job manually and check nvidia-smi to see peak memory usage
#   - Add ~500-1000MB headroom above peak usage for safety
#   - Sentiment (5-class) typically uses more memory than binary tasks
#   - Multitask models use more memory (one head per annotator)
#   - AART uses extra memory for contrastive loss computation
MD_MEMORY_MB=5000                # MD Agreement (binary, smaller dataset)
SENTIMENT_MEMORY_MB=6000         # Sentiment Analysis (5-class, more memory)
HSB_MEMORY_MB=5000               # HSB Binary (binary, similar to MD)
EPIC_MEMORY_MB=6500              # EPIC Dataset

# Fallback if a dataset doesn't have a specific memory setting
DEFAULT_MEMORY_MB=5000

# ─── Retry Configuration (queued mode only) ─────────────────────────────────
# If a job fails with OOM, it can be automatically retried.
# On each retry, the memory requirement increases by RETRY_MEMORY_BOOST_MB
# so the job lands on a less crowded GPU.
# ONLY used by "queued" mode — "parallel" and "sequential" do not retry.
MAX_RETRIES=2                    # How many times to retry a failed job (0=no retries)
RETRY_MEMORY_BOOST_MB=2000      # Extra memory to request on each retry attempt
RETRY_DELAY_SECONDS=30          # Wait this long before retrying a failed job

# ─── Per-Dataset GPU Assignment (parallel mode only) ─────────────────────────
# Each dataset gets exclusive use of its assigned GPU. Pick GPUs with enough
# free memory (check with: nvidia-smi). ONLY used by "parallel" mode.
MD_GPU=0
SENTIMENT_GPU=1
HSB_GPU=2
EPIC_GPU=4

# ─── Single GPU (sequential mode only) ──────────────────────────────────────
# All jobs run on this one GPU, one at a time. ONLY used by "sequential" mode.
SEQ_GPU=0

# ─── Which Datasets to Run ──────────────────────────────────────────────────
RUN_MD=false
RUN_SENTIMENT=false
RUN_HSB=false
RUN_EPIC=true

# ─── Experiment Mode ────────────────────────────────────────────────────────
# "single"     -> uses run_experiments.py (one seed)
# "multi_seed" -> uses run_experiments_multi_seed.py (multiple seeds with stats)
MODE="multi_seed"

# ─── Approaches Per Dataset ─────────────────────────────────────────────────
# Space-separated. Each approach becomes its own job in "queued" mode.
#
# MD / HSB:    majority_vote, aart, aart_rince, multitask,
#              annotator_embedding, annotator_embedding_rince
# Sentiment:   all above + aart_ord_rince, aart_likert
MD_APPROACHES="aart multitask"
SENTIMENT_APPROACHES="aart multitask"
HSB_APPROACHES="aart multitask"
EPIC_APPROACHES="aart multitask"

# ─── Training Configuration ────────────────────────────────────────────────
# Number of training epochs. Leave empty to use the default from each
# dataset's config.py (MD=1, Sentiment=10, HSB=10).
NUM_EPOCHS="10"

# ─── Seeds and Runs (multi_seed mode) ────────────────────────────────────────
SEEDS="42 123"
RUNS_PER_SEED=1

# ─── Noise Configuration ────────────────────────────────────────────────────
# Noise strategy (used for all noisy runs):
#   "fixed"    - All annotators get the same flip probability
#   "random"   - Each annotator gets a random flip prob between 0.1-0.3
#   "custom"   - Use a custom noise_levels dict (set in code)
#   "renegade" - A small % of annotators get high noise, rest get 0%
NOISE_STRATEGY="fixed"

# Noise levels to sweep over (space-separated).
# Each noise level becomes a separate batch of experiments.
# In queued mode, each (dataset, approach, noise_level) triple is its own job.
#
# Use "none" to run WITHOUT noise. You can mix "none" with numeric levels
# to run a clean baseline followed by noisy experiments — all in one launch.
#
# Examples:
#   NOISE_LEVELS="none"                         # No noise at all (clean only)
#   NOISE_LEVELS="0.2"                          # Single noise level
#   NOISE_LEVELS="none 0.1 0.2 0.3"            # Clean baseline + 3 noise levels
#   NOISE_LEVELS="none 0.0 0.1 0.2 0.3 0.4"   # Full sweep including baseline
NOISE_LEVELS="none 0.1 0.2 0.3"

RENEGADE_PERCENT=0.1
RENEGADE_FLIP_PROB=0.7

# ─── Annotator Grouping ─────────────────────────────────────────────────────
USE_GROUPING=false
ANNOTATORS_PER_GROUP=4

# ─── Model Options ──────────────────────────────────────────────────────────
USE_WEIGHTED_EMBEDDINGS=false

# ─── Hyperparameter Overrides (leave empty for defaults) ─────────────────────
LAMBDA2=""
CONTRASTIVE_ALPHA=""
TEMPERATURE=""
RINCE_LAMBDA=""

# RINCE q parameter — controls robustness to noise in RINCE-based approaches
# (aart_rince, annotator_embedding_rince, aart_ord_rince, aart_likert).
# Only has effect when noise is present (ignored for "none" noise levels).
#
# Space-separated list to sweep over multiple values. Each (noise_level, rince_q)
# combination becomes a separate experiment. Leave empty for approach defaults.
#
# Examples:
#   RINCE_Q_VALUES=""                   # Use approach default
#   RINCE_Q_VALUES="0.5"               # Single value for all noisy runs
#   RINCE_Q_VALUES="0.1 0.5 1.0"       # Sweep 3 values per noise level
RINCE_Q_VALUES=""

# ─── Virtual Environment ────────────────────────────────────────────────────
VENV_PATH="venv"


# =============================================================================
# GPU MEMORY UTILITIES
# =============================================================================

get_gpu_free_memory() {
    # Returns free memory in MB for a given GPU ID.
    local gpu_id=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' '
}

get_memory_for_dataset() {
    # Returns the memory requirement (in MB) for a given dataset.
    # Each dataset can have different memory needs based on model size,
    # number of classes, and dataset size.
    local dataset=$1
    case "$dataset" in
        md_agreement)   echo "$MD_MEMORY_MB" ;;
        sentiment)      echo "$SENTIMENT_MEMORY_MB" ;;
        hsb)            echo "$HSB_MEMORY_MB" ;;
        epic)           echo "$EPIC_MEMORY_MB" ;;
        *)              echo "$DEFAULT_MEMORY_MB" ;;
    esac
}

find_gpu_with_memory() {
    # Finds a GPU from the pool with at least the requested free memory.
    # Args: $1 = required memory in MB
    # Returns: GPU ID on success, exits with 1 if none available
    local required_mb=$1
    local gpu_list
    IFS=',' read -ra gpu_list <<< "$GPU_POOL"

    for gpu_id in "${gpu_list[@]}"; do
        local free_mem
        free_mem=$(get_gpu_free_memory "$gpu_id")
        if [ -n "$free_mem" ] && [ "$free_mem" -ge "$required_mb" ]; then
            echo "$gpu_id"
            return 0
        fi
    done
    return 1
}

wait_for_gpu() {
    # Blocks until a GPU with enough free memory is available.
    # Args: $1 = required memory in MB
    # Returns: GPU ID
    local required_mb=$1
    while true; do
        local gpu_id
        if gpu_id=$(find_gpu_with_memory "$required_mb"); then
            echo "$gpu_id"
            return 0
        fi
        sleep "$QUEUE_POLL_INTERVAL"
    done
}

is_oom_error() {
    # Checks if a log file contains out-of-memory error indicators.
    # Args: $1 = path to the job's log file
    # Returns: 0 if OOM detected, 1 otherwise
    local log_file=$1
    if [ ! -f "$log_file" ]; then
        return 1
    fi
    # Check for common CUDA OOM and system OOM messages
    if grep -qiE "CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|RuntimeError.*allocat|torch.cuda.OutOfMemoryError" "$log_file" 2>/dev/null; then
        return 0
    fi
    return 1
}


# =============================================================================
# COMMAND BUILDER
# =============================================================================

build_command() {
    # Builds the full Python command string for one experiment.
    # Args:
    #   $1 = project directory (e.g. "md_agreement_comparison")
    #   $2 = approach (e.g. "aart")
    #   $3 = noise level (e.g. "0.2" or "none" for clean run)
    #   $4 = rince_q value (e.g. "0.5" or "" to use approach default)
    local project_dir=$1
    local approach=$2
    local noise_level=$3
    local rince_q=${4:-""}

    local script_name
    if [ "$MODE" = "multi_seed" ]; then
        script_name="run_experiments_multi_seed.py"
    else
        script_name="run_experiments.py"
    fi

    local script_path="$project_dir/scripts/$script_name"
    if [ ! -f "$script_path" ]; then
        echo "ERROR: Script not found: $script_path" >&2
        return 1
    fi

    local cmd="python $script_path --approaches $approach"

    if [ "$MODE" = "multi_seed" ]; then
        cmd="$cmd --seeds $SEEDS --runs_per_seed $RUNS_PER_SEED"
    fi

    # "none" = clean run (no noise flags). Anything else = noisy run.
    if [ "$noise_level" != "none" ]; then
        cmd="$cmd --add_noise --noise_strategy $NOISE_STRATEGY --noise_level $noise_level"
        if [ "$NOISE_STRATEGY" = "renegade" ]; then
            cmd="$cmd --renegade_percent $RENEGADE_PERCENT --renegade_flip_prob $RENEGADE_FLIP_PROB"
        fi
    fi

    if [ "$USE_GROUPING" = true ]; then
        cmd="$cmd --use_grouping --annotators_per_group $ANNOTATORS_PER_GROUP"
    fi

    if [ "$USE_WEIGHTED_EMBEDDINGS" = true ]; then
        cmd="$cmd --use_weighted_embeddings"
    fi

    [ -n "$LAMBDA2" ]            && cmd="$cmd --lambda2 $LAMBDA2"
    [ -n "$CONTRASTIVE_ALPHA" ]  && cmd="$cmd --contrastive_alpha $CONTRASTIVE_ALPHA"
    [ -n "$TEMPERATURE" ]        && cmd="$cmd --temperature $TEMPERATURE"
    [ -n "$RINCE_LAMBDA" ]       && cmd="$cmd --rince_lambda $RINCE_LAMBDA"
    [ -n "$rince_q" ]            && cmd="$cmd --rince_q $rince_q"
    [ -n "$NUM_EPOCHS" ]         && cmd="$cmd --num_epochs $NUM_EPOCHS"

    echo "$cmd"
}


# =============================================================================
# JOB RUNNERS
# =============================================================================

run_single_job() {
    # Runs one (dataset, approach, noise_level, rince_q) job on a specific GPU.
    # Captures output to a log file for OOM detection.
    # Args:
    #   $1 = dataset name
    #   $2 = approach name
    #   $3 = GPU ID
    #   $4 = log file path
    #   $5 = noise level
    #   $6 = rince_q value (optional, "" to use default)
    local dataset=$1
    local approach=$2
    local gpu_id=$3
    local log_file=$4
    local noise_level=$5
    local rince_q=${6:-""}

    # Map dataset name to project folder
    local project_dir
    case "$dataset" in
        md_agreement)   project_dir="md_agreement_comparison" ;;
        sentiment)      project_dir="sentiment_analysis_comparison" ;;
        hsb)            project_dir="HSB-binary" ;;
        epic)           project_dir="epic_dataset" ;;
    esac

    local cmd
    cmd=$(build_command "$project_dir" "$approach" "$noise_level" "$rince_q")

    local free_mem
    free_mem=$(get_gpu_free_memory "$gpu_id" 2>/dev/null || echo "?")
    local required_mem
    required_mem=$(get_memory_for_dataset "$dataset")

    local job_label=""
    if [ "$noise_level" != "none" ]; then
        job_label=" noise=$noise_level"
    else
        job_label=" (clean)"
    fi
    [ -n "$rince_q" ] && job_label="${job_label} q=$rince_q"

    echo ""
    echo "[$(date '+%H:%M:%S')] START  $dataset/$approach${job_label} -> GPU $gpu_id (${free_mem}MB free, needs ${required_mem}MB)"
    echo "  CMD: CUDA_VISIBLE_DEVICES=$gpu_id $cmd"
    echo "  LOG: $log_file"

    # Run with output captured to log file AND displayed on screen (tee)
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    echo "[$(date '+%H:%M:%S')] DONE   $dataset/$approach${job_label} -> GPU $gpu_id (exit=$exit_code)"
    return $exit_code
}

run_job_with_retry() {
    # Runs a job with automatic OOM retry logic.
    # On each retry:
    #   1. Waits RETRY_DELAY_SECONDS for GPU memory to settle
    #   2. Increases memory requirement by RETRY_MEMORY_BOOST_MB so the
    #      job gets placed on a less crowded GPU
    #   3. Tries again up to MAX_RETRIES times
    #
    # Args:
    #   $1 = dataset name
    #   $2 = approach name
    #   $3 = noise level
    #   $4 = rince_q value (optional)
    local dataset=$1
    local approach=$2
    local noise_level=$3
    local rince_q=${4:-""}

    local base_memory
    base_memory=$(get_memory_for_dataset "$dataset")

    local job_label=""
    if [ "$noise_level" != "none" ]; then
        job_label=" noise=$noise_level"
    else
        job_label=" (clean)"
    fi
    [ -n "$rince_q" ] && job_label="${job_label} q=$rince_q"

    local attempt=0
    local required_memory=$base_memory

    local rince_q_suffix=""
    [ -n "$rince_q" ] && rince_q_suffix="_q${rince_q}"

    while true; do
        attempt=$((attempt + 1))
        local attempt_label="attempt $attempt/$((MAX_RETRIES + 1))"

        # Create a log file for this attempt so we can check for OOM errors
        local log_dir="logs/gpu_scheduler"
        mkdir -p "$log_dir"
        local log_file="$log_dir/${dataset}_${approach}_noise${noise_level}${rince_q_suffix}_attempt${attempt}_$(date '+%Y%m%d_%H%M%S').log"

        echo "[$(date '+%H:%M:%S')] QUEUE  $dataset/$approach${job_label} ($attempt_label, need ${required_memory}MB free)"

        # Wait for a GPU with enough free memory
        local gpu_id
        gpu_id=$(wait_for_gpu "$required_memory")

        # Run the job
        run_single_job "$dataset" "$approach" "$gpu_id" "$log_file" "$noise_level" "$rince_q"
        local exit_code=$?

        # Job succeeded — done!
        if [ $exit_code -eq 0 ]; then
            return 0
        fi

        # Job failed — check if it was an OOM error
        if is_oom_error "$log_file"; then
            echo ""
            echo "[$(date '+%H:%M:%S')] OOM    $dataset/$approach${job_label} failed with out-of-memory on GPU $gpu_id ($attempt_label)"

            # Check if we have retries left
            if [ $attempt -gt $MAX_RETRIES ]; then
                echo "[$(date '+%H:%M:%S')] FAIL   $dataset/$approach${job_label} exhausted all $MAX_RETRIES retries. Giving up."
                return 1
            fi

            # Increase memory requirement for the next attempt so we get
            # placed on a GPU with more free memory (less crowded)
            required_memory=$((required_memory + RETRY_MEMORY_BOOST_MB))
            echo "[$(date '+%H:%M:%S')] RETRY  $dataset/$approach${job_label} will retry in ${RETRY_DELAY_SECONDS}s with ${required_memory}MB requirement"
            sleep "$RETRY_DELAY_SECONDS"
        else
            # Non-OOM failure (e.g., code bug, data issue) — don't retry
            echo "[$(date '+%H:%M:%S')] FAIL   $dataset/$approach${job_label} failed with non-OOM error (exit=$exit_code). Not retrying."
            echo "  Check log: $log_file"
            return $exit_code
        fi
    done
}

run_dataset_all_approaches() {
    # Runs all approaches for a dataset sequentially on one GPU,
    # looping over all noise levels and rince_q values.
    # Used by "parallel" and "sequential" modes.
    # Args:
    #   $1 = dataset name
    #   $2 = GPU ID
    #   $3 = space-separated approaches
    local dataset=$1
    local gpu_id=$2
    local approaches=$3

    local log_dir="logs/gpu_scheduler"
    mkdir -p "$log_dir"

    for noise_level in $NOISE_LEVELS; do
        # Determine rince_q values to sweep for this noise level.
        # Clean runs: rince_q has no effect, so run once with no override.
        # Noisy runs: sweep over RINCE_Q_VALUES if set, otherwise one run with default.
        local rince_q_list=""
        if [ "$noise_level" != "none" ] && [ -n "$RINCE_Q_VALUES" ]; then
            rince_q_list="$RINCE_Q_VALUES"
        fi

        if [ "$noise_level" = "none" ]; then
            echo ""
            echo "---- $dataset: clean (no noise) ----"
        else
            echo ""
            echo "---- $dataset: noise_level=$noise_level ----"
        fi

        for approach in $approaches; do
            if [ -n "$rince_q_list" ]; then
                for rince_q in $rince_q_list; do
                    echo "  >> rince_q=$rince_q"
                    local log_file="$log_dir/${dataset}_${approach}_noise${noise_level}_q${rince_q}_$(date '+%Y%m%d_%H%M%S').log"
                    run_single_job "$dataset" "$approach" "$gpu_id" "$log_file" "$noise_level" "$rince_q"
                done
            else
                local log_file="$log_dir/${dataset}_${approach}_noise${noise_level}_$(date '+%Y%m%d_%H%M%S').log"
                run_single_job "$dataset" "$approach" "$gpu_id" "$log_file" "$noise_level" ""
            fi
        done
    done
}


# =============================================================================
# MAIN
# =============================================================================

# Activate virtual environment
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated virtualenv: $VENV_PATH"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Count total jobs.
# For each noise level: if noisy and RINCE_Q_VALUES is set, multiply by rince_q count.
# For "none" noise levels, rince_q has no effect so count = 1.
noise_count=$(echo $NOISE_LEVELS | wc -w)
rince_q_count=$(echo $RINCE_Q_VALUES | wc -w)
noisy_count=0
clean_count=0
for nl in $NOISE_LEVELS; do
    if [ "$nl" = "none" ]; then
        clean_count=$((clean_count + 1))
    else
        noisy_count=$((noisy_count + 1))
    fi
done

# Each noisy level gets multiplied by rince_q sweep (or 1 if no sweep)
rince_q_mult=$rince_q_count
[ "$rince_q_mult" -eq 0 ] && rince_q_mult=1
configs_per_approach=$((clean_count + noisy_count * rince_q_mult))

total_jobs=0
[ "$RUN_MD" = true ]        && total_jobs=$((total_jobs + $(echo $MD_APPROACHES | wc -w) * configs_per_approach))
[ "$RUN_SENTIMENT" = true ] && total_jobs=$((total_jobs + $(echo $SENTIMENT_APPROACHES | wc -w) * configs_per_approach))
[ "$RUN_HSB" = true ]       && total_jobs=$((total_jobs + $(echo $HSB_APPROACHES | wc -w) * configs_per_approach))
[ "$RUN_EPIC" = true ]      && total_jobs=$((total_jobs + $(echo $EPIC_APPROACHES | wc -w) * configs_per_approach))

echo ""
echo "============================================================"
echo "  Experiment Launcher"
echo "  Exec mode:        $EXEC_MODE"
echo "  Experiment type:   $MODE"
echo "  Datasets:          MD=$RUN_MD  Sentiment=$RUN_SENTIMENT  HSB=$RUN_HSB  EPIC=$RUN_EPIC"
echo "  GPU pool:          $GPU_POOL"
echo "  Memory per job:    MD=${MD_MEMORY_MB}MB  Sent=${SENTIMENT_MEMORY_MB}MB  HSB=${HSB_MEMORY_MB}MB  EPIC=${EPIC_MEMORY_MB}MB"
echo "  Retry policy:      max_retries=$MAX_RETRIES, boost=${RETRY_MEMORY_BOOST_MB}MB/retry"
echo "  Noise configs:     $NOISE_LEVELS ($noise_count configurations)"
if [ -n "$RINCE_Q_VALUES" ]; then
echo "  RINCE q sweep:     $RINCE_Q_VALUES ($rince_q_count values, noisy runs only)"
fi
echo "  Total jobs:        $total_jobs"
echo "  Started:           $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Show current GPU status
echo ""
echo "Current GPU status:"
nvidia-smi --query-gpu=index,name,memory.free,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null | while read -r line; do
    echo "  GPU $line"
done
echo ""

PIDS=()
NAMES=()

case "$EXEC_MODE" in

    # ── PARALLEL MODE ────────────────────────────────────────────────────────
    # Each dataset gets a dedicated GPU. All approaches for that dataset run
    # sequentially on its assigned GPU. Datasets run simultaneously.
    parallel)
        if [ "$RUN_MD" = true ] && [ "$MD_GPU" -ge 0 ]; then
            run_dataset_all_approaches "md_agreement" "$MD_GPU" "$MD_APPROACHES" &
            PIDS+=($!)
            NAMES+=("md_agreement")
        fi
        if [ "$RUN_SENTIMENT" = true ] && [ "$SENTIMENT_GPU" -ge 0 ]; then
            run_dataset_all_approaches "sentiment" "$SENTIMENT_GPU" "$SENTIMENT_APPROACHES" &
            PIDS+=($!)
            NAMES+=("sentiment")
        fi
        if [ "$RUN_HSB" = true ] && [ "$HSB_GPU" -ge 0 ]; then
            run_dataset_all_approaches "hsb" "$HSB_GPU" "$HSB_APPROACHES" &
            PIDS+=($!)
            NAMES+=("hsb")
        fi
        if [ "$RUN_EPIC" = true ] && [ "$EPIC_GPU" -ge 0 ]; then
            run_dataset_all_approaches "epic" "$EPIC_GPU" "$EPIC_APPROACHES" &
            PIDS+=($!)
            NAMES+=("epic")
        fi

        echo "Launched ${#PIDS[@]} dataset groups in parallel. Waiting..."
        for i in "${!PIDS[@]}"; do
            wait "${PIDS[$i]}" 2>/dev/null
            echo "  [${NAMES[$i]}] Done (exit=$?)"
        done
        ;;

    # ── SEQUENTIAL MODE ──────────────────────────────────────────────────────
    # All jobs on one GPU, one at a time.
    sequential)
        if [ "$RUN_MD" = true ]; then
            run_dataset_all_approaches "md_agreement" "$SEQ_GPU" "$MD_APPROACHES"
        fi
        if [ "$RUN_SENTIMENT" = true ]; then
            run_dataset_all_approaches "sentiment" "$SEQ_GPU" "$SENTIMENT_APPROACHES"
        fi
        if [ "$RUN_HSB" = true ]; then
            run_dataset_all_approaches "hsb" "$SEQ_GPU" "$HSB_APPROACHES"
        fi
        if [ "$RUN_EPIC" = true ]; then
            run_dataset_all_approaches "epic" "$SEQ_GPU" "$EPIC_APPROACHES"
        fi
        ;;

    # ── QUEUED MODE (memory-aware with OOM retry) ────────────────────────────
    # Each (dataset, approach) pair is an independent job. Jobs check GPU
    # memory via nvidia-smi and pick a GPU with enough free memory.
    # Multiple jobs CAN share a GPU if each job fits within the free memory.
    #
    # Memory requirements are per-dataset:
    #   - MD needs MD_MEMORY_MB (default 5000MB)
    #   - Sentiment needs SENTIMENT_MEMORY_MB (default 6000MB)
    #   - HSB needs HSB_MEMORY_MB (default 5000MB)
    #
    # OOM retry behavior:
    #   - If a job fails with CUDA OOM, it waits RETRY_DELAY_SECONDS
    #   - On retry, it requests RETRY_MEMORY_BOOST_MB more free memory
    #     so it lands on a less crowded GPU
    #   - Retries up to MAX_RETRIES times before giving up
    #   - Non-OOM failures (code bugs, data errors) are NOT retried
    queued)
        # In queued mode, each (dataset, approach, noise_level, rince_q) combo
        # becomes an independent job that competes for GPU time.
        # "none" = clean run (no noise flags passed to Python).
        # rince_q sweep only applies to noisy runs.

        # Helper to launch jobs for one (noise_level, rince_q) combination
        launch_queued_jobs() {
            local noise_level=$1
            local rince_q=$2

            local level_tag="n=$noise_level"
            [ "$noise_level" = "none" ] && level_tag="clean"
            [ -n "$rince_q" ] && level_tag="${level_tag}/q=$rince_q"

            if [ "$RUN_MD" = true ]; then
                for approach in $MD_APPROACHES; do
                    run_job_with_retry "md_agreement" "$approach" "$noise_level" "$rince_q" &
                    PIDS+=($!)
                    NAMES+=("md/$approach/$level_tag")
                    sleep 2
                done
            fi
            if [ "$RUN_SENTIMENT" = true ]; then
                for approach in $SENTIMENT_APPROACHES; do
                    run_job_with_retry "sentiment" "$approach" "$noise_level" "$rince_q" &
                    PIDS+=($!)
                    NAMES+=("sent/$approach/$level_tag")
                    sleep 2
                done
            fi
            if [ "$RUN_HSB" = true ]; then
                for approach in $HSB_APPROACHES; do
                    run_job_with_retry "hsb" "$approach" "$noise_level" "$rince_q" &
                    PIDS+=($!)
                    NAMES+=("hsb/$approach/$level_tag")
                    sleep 2
                done
            fi
            if [ "$RUN_EPIC" = true ]; then
                for approach in $EPIC_APPROACHES; do
                    run_job_with_retry "epic" "$approach" "$noise_level" "$rince_q" &
                    PIDS+=($!)
                    NAMES+=("epic/$approach/$level_tag")
                    sleep 2
                done
            fi
        }

        for noise_level in $NOISE_LEVELS; do
            if [ "$noise_level" = "none" ]; then
                # Clean run: no rince_q sweep
                launch_queued_jobs "$noise_level" ""
            elif [ -n "$RINCE_Q_VALUES" ]; then
                # Noisy run with rince_q sweep
                for rince_q in $RINCE_Q_VALUES; do
                    launch_queued_jobs "$noise_level" "$rince_q"
                done
            else
                # Noisy run without rince_q sweep
                launch_queued_jobs "$noise_level" ""
            fi
        done

        echo ""
        echo "Launched ${#PIDS[@]} jobs in queued mode. Waiting for all to complete..."
        echo "(Jobs will auto-assign to GPUs based on per-dataset memory needs)"
        echo "(Failed OOM jobs will retry up to $MAX_RETRIES times with +${RETRY_MEMORY_BOOST_MB}MB)"
        echo ""

        # Wait for all jobs and report results
        failed=0
        for i in "${!PIDS[@]}"; do
            if wait "${PIDS[$i]}" 2>/dev/null; then
                echo "  [${NAMES[$i]}] PASSED"
            else
                echo "  [${NAMES[$i]}] FAILED"
                failed=$((failed + 1))
            fi
        done

        if [ "$failed" -gt 0 ]; then
            echo ""
            echo "WARNING: $failed job(s) failed. Check logs in logs/gpu_scheduler/"
        fi
        ;;

    *)
        echo "ERROR: Unknown EXEC_MODE '$EXEC_MODE'. Use: parallel | sequential | queued"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  All experiments complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
