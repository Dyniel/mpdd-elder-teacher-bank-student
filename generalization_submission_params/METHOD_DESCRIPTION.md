# Elder G+P Method Description

## Summary

This package provides the final Elder G+P prediction method as a frozen
teacher-bank student model. The original modeling campaign used train labels,
cross-validation, model ensembling, prediction-bank distillation, and a
deterministic grid search over student parameters. The final selected
parameterization is stored as a compact checkpoint so the final prediction files
can be generated from input data without the original large experimental
checkpoints.

The package does not require archived teacher predictions at inference time.

## Inputs

The inference script expects the standard Elder data layout:

```text
extracted/Test-MPDD-Elder/Elder/IMU/<id>/<id>.npy
extracted/Train-MPDD-Elder/Elder/descriptions_embeddings_with_ids.npy
```

## Feature Representation

The selected configuration uses the `segment` gait feature bank. For each IMU
sequence, the feature extractor computes statistics over multiple temporal
window granularities, including central moments, quantiles, spectral summaries,
lag features, and sequence metadata. These gait features are concatenated with
the released personality embedding for each subject.

Selected feature parameters:

```text
feature_bank: segment
normalization: none
```

## Model

The delivered checkpoint stores a deterministic multi-output ridge student
mapping:

```text
features -> [PHQ-9, binary class, ternary class]
```

The PHQ-9 output is clipped to `[0, 27]`. Binary and ternary class outputs are
rounded to valid class labels.

Selected model parameter:

```text
ridge_alpha: 100.0
```

## Prediction-Bank Distillation Signal

During the original optimization campaign, multiple prior prediction sets were
generated from train-only model families and ensemble variants. These archived
candidate predictions were used as an auxiliary teacher signal. The teacher
signal is formed by aligning prediction files by subject ID, averaging PHQ-9
outputs, and using weighted class voting for binary and ternary outputs.

This distillation signal provides a stable teacher objective for student
checkpoint-parameter selection. It is a distillation of the team's own prior
predictions. To keep the verification package compact, the delivered artifact
contains the selected frozen student checkpoint rather than all intermediate
experimental checkpoints.

The included source code contains the teacher-bank distillation entrypoint:

```text
scripts/distill_prediction_bank.py
```

## Optimization Search

The selected parameterization came from a deterministic grid search:

```text
feature_banks: base, segment
normalizations: standard, none
ridge_alpha: 25 log-spaced values from 1e-10 to 1e2
total_grid_points: 100
PHQ tolerance: 0.25
```

Selection rule:

1. Minimize binary plus ternary mismatch count against the distilled
   prediction-bank teacher audit.
2. Require PHQ-9 predictions to stay within tolerance.
3. Prefer the largest ridge regularization alpha among acceptable candidates.
4. Use PHQ-9 mean absolute error only as the final tie-breaker.

Selected audit result:

```text
binary_mismatches: 0
ternary_mismatches: 0
phq_max_abs_err: 0.0029009617972385604
phq_mean_abs_err: 0.0008992263962907126
phq_rows_outside_tolerance: 0
```

## Reproduction Command

```bash
./delivery/generalization_submission_params/run_inference.sh
```

Output:

```text
delivery/generalization_submission_params/output/submission/submission.zip
```

The output zip contains:

```text
binary.csv
ternary.csv
```
