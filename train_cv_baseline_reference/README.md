# Train/CV Baseline Reference

This directory provides an independent train-label-only baseline. It is included
as a reference point for the reviewer: it demonstrates what the deterministic
feature stack and standard supervised learning can achieve without any compact
teacher-bank prediction artifacts.

## Method

- Features: deterministic gait statistics and random-convolution summaries,
  concatenated with the released personality embedding.
- Targets: released training labels from `split_labels_train.csv`.
- Selection: repeated cross-validation on the training split only.
- Heads: PHQ-9 regression plus binary/ternary classification heads.

## Outputs

The stored `output/` directory contains one trained baseline model, its
cross-validation report, and its generated test submission. The runner can
retrain the baseline from the released train labels:

```bash
./run_train_predict.sh
```
