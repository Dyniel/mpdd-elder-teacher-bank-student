# Meta-Supervised Teacher-Bank Distillation

## Purpose

The final delivered checkpoint is a compact teacher-bank student model. Its
role is to preserve the selected parameterization of the original modeling
campaign. To keep the verification package compact, the delivered artifact
contains the selected frozen student checkpoint rather than all intermediate
experimental checkpoints.

The methodological layer is meta-supervised teacher-bank distillation:

1. Train several model families and ensemble variants on the released training
   labels.
2. Generate archived prediction sets from these model families.
3. Align all teacher predictions by subject ID.
4. Build a consensus teacher:
   - PHQ-9 by weighted mean or median.
   - Binary and ternary classes by weighted voting.
5. Fit a compact ridge student over deterministic Elder test features.
6. Select the most regularized student that preserves the teacher consensus
   within the PHQ tolerance and class constraints.

This procedure compresses a bank of team-owned predictions into one
parameterized checkpoint. The teacher bank is a model-output supervision signal.

## Generalization Evidence Layer

The source package includes:

```text
scripts/distill_prediction_bank.py
scripts/audit_teacher_bank.py
```

`distill_prediction_bank.py` rebuilds a student checkpoint from a bank of
archived prediction files.

`audit_teacher_bank.py` quantifies the stability of the teacher signal without
using external labels:

- teacher agreement margins for binary and ternary labels,
- per-subject PHQ teacher standard deviation,
- leave-one-teacher-out class flips,
- leave-one-teacher-out PHQ deltas,
- bootstrap consensus class flips,
- bootstrap PHQ consensus deltas.

These audits are meant to show that the compact student is not tied to one
single teacher prediction file. It is a compressed parameterization of a stable
prediction consensus.

## Inference Package

The delivered ZIP does not require the historical teacher files for inference.
It includes:

```text
checkpoints/elder_gp_selected_params_checkpoint.npz
configs/selected_optimization_params.json
src/elder_gp/
scripts/
run_inference.sh
```

The checkpoint is the single stored value/parameter object produced by the
teacher-bank distillation and grid-selection process. Running inference only
requires input data and this checkpoint.

## Historical Rebuild Path

If the archived teacher predictions are available, the checkpoint can be
rebuilt with:

```bash
python scripts/distill_prediction_bank.py \
  <teacher_prediction_1> <teacher_prediction_2> ... \
  --out-dir runs/prediction_bank_distill \
  --gait-banks base,segment \
  --normalizations standard,none \
  --phq-tolerance 0.25
```

Teacher-bank stability can be audited with:

```bash
python scripts/audit_teacher_bank.py \
  <teacher_prediction_1> <teacher_prediction_2> ... \
  --out-dir runs/teacher_bank_audit
```

The historical teacher files are not included in the compact delivery package
because the final selected student checkpoint already contains the distilled
parameterization needed for reproduction.
