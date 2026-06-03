# Deliverables

## Method Description

```text
delivery/generalization_submission_params/METHOD_DESCRIPTION.md
delivery/generalization_submission_params/TEACHER_BANK_METHODOLOGY.md
```

## Source Code

Full pipeline source code is included:

```text
delivery/generalization_submission_params/src/elder_gp/
delivery/generalization_submission_params/scripts/
```

This includes the deterministic feature extractor, train-only feature stack,
neural/foundation model entrypoints, ternary/binary heads, and related utility
modules. It also includes:

```text
delivery/generalization_submission_params/scripts/distill_prediction_bank.py
delivery/generalization_submission_params/scripts/audit_teacher_bank.py
```

These entrypoints rebuild and audit the teacher-bank distillation path when
archived teacher predictions are available. Inference uses the compact student
checkpoint listed below; the larger experimental checkpoints from the original
modeling campaign are not included.

## Trained Checkpoint / Model Weights

```text
delivery/generalization_submission_params/checkpoints/elder_gp_selected_params_checkpoint.npz
```

## Configuration

```text
delivery/generalization_submission_params/configs/selected_optimization_params.json
delivery/generalization_submission_params/requirements.txt
delivery/generalization_submission_params/requirements-elder-gp.txt
delivery/generalization_submission_params/requirements-elder-gp-gpu.txt
```

The config records the selected feature bank, normalization, ridge alpha,
grid-search space, iteration budget, and selected audit metrics.

## Run Script

```text
delivery/generalization_submission_params/run_inference.sh
```

Run:

```bash
./delivery/generalization_submission_params/run_inference.sh
```

Expected output:

```text
delivery/generalization_submission_params/output/submission/submission.zip
```
