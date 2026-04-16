# Renegade Annotator Experiments Status

## Summary

✅ **All renegade annotator experiments are now fixed and ready to run across MDA, HSB, and SNT datasets.**

## What Was Fixed

The renegade noise strategy was not working correctly because:
1. `create_noise_config` was never being called in the training scripts
2. When it was called, it used integer indices (0, 1, 2, ...) but the actual annotator IDs are strings (e.g., "annotator_1", "R_1I5depz6ASpP2YF")
3. The noise_levels dictionary needed to map from actual annotator ID strings to noise levels

### Fixes Applied

**All three datasets (HSB, MDA, SNT):**
- Updated `train.py` to properly call `create_noise_config` for renegade strategy
- Added mapping from integer indices to actual annotator ID strings
- Ensured renegade parameters (`renegade_percent`, `renegade_flip_prob`) are properly passed from config

## How Renegade Strategy Works

1. **Selects renegades**: Randomly selects a percentage of annotators (default 10%) to be "renegades"
2. **High noise for renegades**: Renegade annotators have a high probability (default 70%) of flipping labels
3. **Zero noise for others**: Non-renegade annotators have 0% noise

## Running Renegade Experiments

### Single Seed (for testing)

**HSB:**
```bash
cd HSB-binary
python scripts/run_experiments.py \
  --approaches majority_vote aart multitask annotator_embedding \
  --add_noise \
  --noise_strategy renegade \
  --renegade_percent 0.1 \
  --renegade_flip_prob 0.7
```

**MDA:**
```bash
cd md_agreement_comparison
python scripts/run_experiments.py \
  --approaches majority_vote aart multitask annotator_embedding \
  --add_noise \
  --noise_strategy renegade \
  --renegade_percent 0.1 \
  --renegade_flip_prob 0.7
```

**SNT:**
```bash
cd sentiment_analysis_comparison
python scripts/run_experiments.py \
  --approaches majority_vote aart multitask annotator_embedding \
  --add_noise \
  --noise_strategy renegade \
  --renegade_percent 0.1 \
  --renegade_flip_prob 0.7
```

### Multi-Seed (for production)

**HSB:**
```bash
cd HSB-binary
python scripts/run_experiments_multi_seed.py \
  --approaches majority_vote aart multitask annotator_embedding \
  --seeds 42 123 456 789 1011 \
  --add_noise \
  --noise_strategy renegade \
  --renegade_percent 0.1 \
  --renegade_flip_prob 0.7
```

**MDA:**
```bash
cd md_agreement_comparison
python scripts/run_experiments_multi_seed.py \
  --approaches majority_vote aart multitask annotator_embedding \
  --seeds 42 123 456 789 1011 \
  --add_noise \
  --noise_strategy renegade \
  --renegade_percent 0.1 \
  --renegade_flip_prob 0.7
```

**SNT:**
```bash
cd sentiment_analysis_comparison
python scripts/run_experiments_multi_seed.py \
  --approaches majority_vote aart multitask annotator_embedding \
  --seeds 42 123 456 789 1011 \
  --add_noise \
  --noise_strategy renegade \
  --renegade_percent 0.1 \
  --renegade_flip_prob 0.7
```

## Parameters

- `--renegade_percent`: Percentage of annotators to be renegades (default: 0.1 = 10%)
- `--renegade_flip_prob`: Probability of flipping labels for renegade annotators (default: 0.7 = 70%)
- `--noise_strategy renegade`: Must be set to use renegade strategy
- `--add_noise`: Must be enabled for noise to be applied

## Verification

All three datasets now:
- ✅ Properly create noise config for renegade strategy
- ✅ Map integer indices to actual annotator ID strings
- ✅ Pass renegade parameters through config to training
- ✅ Support multi-seed experiments with renegade strategy
- ✅ Log renegade annotator selection and noise application

## Files Modified

1. `HSB-binary/scripts/train.py` - Fixed renegade noise config creation
2. `md_agreement_comparison/scripts/train.py` - Fixed renegade noise config creation
3. `sentiment_analysis_comparison/scripts/train.py` - Fixed renegade noise config creation

## Notes

- The renegade selection is randomized, so different runs will select different annotators as renegades
- For reproducible results, ensure seeds are set (multi-seed scripts handle this automatically)
- Renegade strategy is particularly useful for testing model robustness to adversarial annotators

