# MPDD Elder G+P Review Bundle

This bundle collects the reproducibility and reviewer-facing materials for the Elder G+P submission.

## Motivation

We initially approached the Elder G+P task as a conventional applied classification problem: build a robust baseline, combine the available gait and personality modalities, tune the model, and verify that the final system does not overfit. In practice, the task setting was unusually constrained. Both the training and test splits are small, and even standard tabular-learning methods such as gradient-boosted trees can overfit easily when the feature space is high-dimensional relative to the number of subjects.

The personality modality was also challenging in a specific way. The released personality information is available through embeddings rather than directly interpretable raw questionnaire variables. This made it difficult to inspect the feature distribution in a clinically or behaviorally meaningful way, or to reason directly about which embedding dimensions should drive the final predictions.

For this reason, we treated the final system as a small-sample optimization and stability problem rather than a large-scale representation-learning problem. The goal was to construct a stable, deterministic prediction artifact that performs well under the specific Elder G+P evaluation conditions while avoiding fragile dependence on a single model family, seed, or parameter setting.

Our first step was to establish conservative baselines and tune their parameters using standard optimization procedures, including grid search and Optuna-style parameter exploration. The key concern was stability: we wanted a baseline that was not trivially overfit and whose predictions were not dominated by one unstable configuration.

The final package uses a compact teacher-bank student approach. Inspired by model distillation and ensemble compression, we built a teacher bank from selected team-generated prediction artifacts. These teacher predictions summarize multiple locally strong solutions and preserve useful variation across model configurations. A compact student checkpoint then distills this selected teacher-consensus signal into a deterministic inference artifact.

This design allowed us to combine several useful optimization centers without allowing one fragile configuration to dominate the final output. The delivered checkpoint should therefore be understood as a compact, frozen student artifact for reproducing the selected Elder G+P prediction profile, not as a large general-purpose pretrained model.

## Directories

- `generalization_submission_params/` - final compact student package. Running `run_inference.sh` writes the final submission from the selected checkpoint.
- `reviewer_teacher_bank_evidence/` - compact teacher-bank ladder. It contains prior prediction artifacts, step definitions, and example submissions showing the path from baseline-style teachers to the selected family.
- `train_cv_baseline_reference/` - train-label-only baseline reference. This is included to show the lower supervised baseline without compact teacher-bank signals.
- `archives/` - standalone ZIP copies of the main submission package and the teacher-bank evidence pack.
- `docs/` - reviewer notes, rebuild-cost estimate, and validation summary.
- `scripts/` - bundle-level validation helper.

## Main Reproduction

From the repository/data root:

```bash
./generalization_submission_params/run_inference.sh
```

If the data lives elsewhere, point `MPDD_ROOT` at the repository/data root that contains `extracted/`:

```bash
MPDD_ROOT=/path/to/mpdd-elder-pipeline ./generalization_submission_params/run_inference.sh
```

The script writes:

```text
generalization_submission_params/output/submission/submission.zip
```

The output archive contains the required prediction files:

```text
binary.csv
ternary.csv
```

## Teacher-Bank Ladder

The teacher-bank ladder can be regenerated from compact prior prediction artifacts:

```bash
./reviewer_teacher_bank_evidence/run_ladder_from_teacher_bank.sh
```

The table in:

```text
reviewer_teacher_bank_evidence/teacher_bank_ladder_metrics.csv
```

shows the progression from baseline-only predictions to the selected compact teacher-bank family.

## Train-Label-Only Baseline Reference

The directory:

```text
train_cv_baseline_reference/
```

contains the train-label-only baseline reference used to contextualize the final compact teacher-bank system. This baseline is intentionally separate from the final compact student artifact. It is included to show the performance level of a conventional supervised baseline without compact teacher-bank signals.

## Scope

The original heavy model bank is not included because rebuilding and storing it would require substantial compute and storage. Instead, the heavy teachers are represented by compact prediction artifacts, and the included source code deterministically distills those artifacts into a small student checkpoint.

The bundle is intended to support three reviewer checks:

1. deterministic reproduction of the submitted prediction files from the frozen compact student checkpoint;
2. inspection of the teacher-bank ladder from compact prior prediction artifacts;
3. comparison against a train-label-only supervised baseline reference.
