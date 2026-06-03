"""Train the baseline Elder G+P model on the released train labels and predict test.

No teacher bank, no test-set fitting, no ID-locked checkpoint. All model
selection is done by cross-validation on the training split only.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from elder_gp_baseline.baseline_model import (
    _oof_classify,
    build_tables,
    cv_report,
    predict_test,
    select_and_fit,
)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_submission(out_dir: Path, ids, phq, binary, ternary) -> Path:
    sub = out_dir / "submission"
    _write_csv(
        sub / "binary.csv",
        [
            {"id": int(i), "binary_pred": int(b), "phq9_pred": f"{p:.8f}"}
            for i, b, p in zip(ids, binary, phq, strict=True)
        ],
        ["id", "binary_pred", "phq9_pred"],
    )
    _write_csv(
        sub / "ternary.csv",
        [
            {"id": int(i), "ternary_pred": int(t), "phq9_pred": f"{p:.8f}"}
            for i, t, p in zip(ids, ternary, phq, strict=True)
        ],
        ["id", "ternary_pred", "phq9_pred"],
    )
    zip_path = sub / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(sub / "binary.csv", arcname="binary.csv")
        archive.write(sub / "ternary.csv", arcname="ternary.csv")
    return zip_path


def _baseline_f1(x, y_ternary) -> dict:
    """Majority-class baseline so the CV numbers have an baseline reference point."""
    f1 = _oof_classify(DummyClassifier(strategy="most_frequent"), x, y_ternary)
    return {"ternary_macro_f1": float(f1[0]), "binary_macro_f1": float(f1[1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default="extracted/Train-MPDD-Elder/Elder")
    parser.add_argument("--test-root", default="extracted/Test-MPDD-Elder/Elder")
    parser.add_argument("--label-csv", default="extracted/Train-MPDD-Elder/Elder/split_labels_train.csv")
    parser.add_argument(
        "--personality-npy",
        default="extracted/Train-MPDD-Elder/Elder/descriptions_embeddings_with_ids.npy",
    )
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()

    train_root = Path(args.train_root).resolve()
    test_root = Path(args.test_root).resolve()
    label_csv = Path(args.label_csv).resolve()
    personality_npy = Path(args.personality_npy).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[baseline] extracting features (gait banks: base, segment) ...")
    train_tables, test_tables = build_tables(train_root, test_root, label_csv, personality_npy)

    any_train = next(iter(train_tables.values()))
    y_ternary = any_train.label3.astype(np.int64)
    y_phq = any_train.phq9.astype(np.float64)

    print("[baseline] cross-validating candidates on the training split ...")
    model, candidates = select_and_fit(train_tables, y_ternary, y_phq)

    report = cv_report(model, candidates)
    from elder_gp_baseline.baseline_model import combined_matrix

    report["majority_baseline_cv"] = _baseline_f1(combined_matrix(train_tables[model.gait_bank]), y_ternary)

    print("[baseline] predicting test set ...")
    pred = predict_test(model, test_tables[model.gait_bank])

    zip_path = _write_submission(out_dir, pred["ids"], pred["phq9"], pred["binary"], pred["ternary"])

    _write_csv(
        out_dir / "selected_test_predictions.csv",
        [
            {"id": int(i), "binary_pred": int(b), "ternary_pred": int(t), "phq9_pred": float(p)}
            for i, b, t, p in zip(pred["ids"], pred["binary"], pred["ternary"], pred["phq9"], strict=True)
        ],
        ["id", "binary_pred", "ternary_pred", "phq9_pred"],
    )
    (out_dir / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "gait_bank": model.gait_bank,
            "ternary_strategy": model.ternary.strategy,
            "ternary_estimator": model.ternary_estimator,
            "phq_estimator": model.phq_estimator,
            "phq_band_edges": model.metadata["phq_band_edges"],
        },
        out_dir / "baseline_model.joblib",
    )

    print(json.dumps(report["selected"], indent=2))
    print(json.dumps(report["majority_baseline_cv"], indent=2))
    print(f"[submission] {zip_path}")
    print(f"[cv_report]  {out_dir / 'cv_report.json'}")


if __name__ == "__main__":
    main()
