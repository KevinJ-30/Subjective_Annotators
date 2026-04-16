# Number of Crowd-Workers per Sample (#A/E)

This document summarizes the number of annotators per sample for each dataset.

## Summary

| Dataset | #A/E (Annotators per Example) | Notes |
|---------|-------------------------------|-------|
| **MDA** (Multi-Domain Agreement) | **5** | Consistent across all samples |
| **HSB** (Hate Speech Binary) | **6** | Consistent across train and test |
| **SNT** (Sentiment Analysis) | **4** (mostly) | Variable: 4-12 in train, 4-6 in test |

## Detailed Analysis

### MDA (Multi-Domain Agreement)
- **#A/E: 5**
- Consistent across all samples

### HSB (Hate Speech Binary)
- **#A/E: 6**
- **Train set**: 784 samples, all with 6 annotators
- **Test set**: 168 samples, all with 6 annotators
- Consistent across train and test splits

### SNT (Sentiment Analysis)
- **#A/E: 4** (primary, but variable)
- **Train set**: 14,071 samples
  - 4 annotators: 11,369 samples (80.8%)
  - 5 annotators: 2,577 samples (18.3%)
  - 6 annotators: 79 samples (0.6%)
  - 8 annotators: 27 samples (0.2%)
  - 9 annotators: 10 samples (0.1%)
  - 10 annotators: 7 samples (<0.1%)
  - 12 annotators: 2 samples (<0.1%)
- **Test set**: 338 samples
  - 4 annotators: 272 samples (80.5%)
  - 5 annotators: 65 samples (19.2%)
  - 6 annotators: 1 sample (0.3%)

## Analysis Scripts

Scripts to verify these counts are available in each dataset's processed folder:
- `HSB-binary/data/hsb_brexit/processed/count_annotators.py`
- `sentiment_analysis_comparison/data/sentiment_analysis/processed/count_annotators.py`

Run these scripts to regenerate the analysis:
```bash
cd HSB-binary/data/hsb_brexit/processed && python3 count_annotators.py
cd sentiment_analysis_comparison/data/sentiment_analysis/processed && python3 count_annotators.py
```

