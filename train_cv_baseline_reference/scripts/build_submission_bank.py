"""Build an baseline submission bank -> consensus -> final submission.

Every bank member is trained only on the released training labels. The
consensus (weighted vote for classes, weighted mean for PHQ-9) is validated
out-of-fold on the training split and compared against the best single member.
The test set is never used to fit, weight, or select anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np

from elder_gp_baseline.baseline_model import build_tables, combined_matrix
from elder_gp_baseline.submission_bank import (
    classifier_bank,
    fit_bank,
    oof_phq_consensus,
    oof_ternary_consensus,
    predict_bank,
    regressor_bank,
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
        [{"id": int(i), "binary_pred": int(b), "phq9_pred": f"{p:.8f}"} for i, b, p in zip(ids, binary, phq, strict=True)],
        ["id", "binary_pred", "phq9_pred"],
    )
    _write_csv(
        sub / "ternary.csv",
        [{"id": int(i), "ternary_pred": int(t), "phq9_pred": f"{p:.8f}"} for i, t, p in zip(ids, ternary, phq, strict=True)],
        ["id", "ternary_pred", "phq9_pred"],
    )
    zip_path = sub / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(sub / "binary.csv", arcname="binary.csv")
        archive.write(sub / "ternary.csv", arcname="ternary.csv")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default="extracted/Train-MPDD-Elder/Elder")
    parser.add_argument("--test-root", default="extracted/Test-MPDD-Elder/Elder")
    parser.add_argument("--label-csv", default="extracted/Train-MPDD-Elder/Elder/split_labels_train.csv")
    parser.add_argument("--personality-npy", default="extracted/Train-MPDD-Elder/Elder/descriptions_embeddings_with_ids.npy")
    parser.add_argument("--gait-bank", default="base", choices=("base", "segment"))
    parser.add_argument("--out-dir", default="output/bank")
    args = parser.parse_args()

    train_root = Path(args.train_root).resolve()
    test_root = Path(args.test_root).resolve()
    label_csv = Path(args.label_csv).resolve()
    personality_npy = Path(args.personality_npy).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bank] extracting features (gait bank: {args.gait_bank}) ...")
    train_tables, test_tables = build_tables(train_root, test_root, label_csv, personality_npy)
    train_table = train_tables[args.gait_bank]
    test_table = test_tables[args.gait_bank]

    x_train = combined_matrix(train_table)
    y_ternary = train_table.label3.astype(np.int64)
    y_phq = train_table.phq9.astype(np.float64)

    clf = classifier_bank()
    reg = regressor_bank()

    print(f"[bank] OOF-validating ternary consensus ({len(clf)} classifiers) ...")
    cons_tern_f1, cons_bin_f1, member_tern_f1 = oof_ternary_consensus(x_train, y_ternary, clf)
    print(f"[bank] OOF-validating PHQ consensus ({len(reg)} regressors) ...")
    cons_mae, member_mae = oof_phq_consensus(x_train, y_phq, y_ternary, reg)

    best_member_tern = max(member_tern_f1.values())
    best_member_mae = min(member_mae.values())

    print("[bank] refitting full bank and predicting test ...")
    fitted = fit_bank(x_train, y_ternary, y_phq)
    pred = predict_bank(fitted, combined_matrix(test_table))

    zip_path = _write_submission(out_dir, test_table.ids, pred["phq9"], pred["binary"], pred["ternary"])
    _write_csv(
        out_dir / "selected_test_predictions.csv",
        [
            {"id": int(i), "binary_pred": int(b), "ternary_pred": int(t), "phq9_pred": float(p)}
            for i, b, t, p in zip(test_table.ids, pred["binary"], pred["ternary"], pred["phq9"], strict=True)
        ],
        ["id", "binary_pred", "ternary_pred", "phq9_pred"],
    )

    report = {
        "method": "baseline submission bank -> consensus (weighted vote + mean), train-only, OOF-validated",
        "gait_bank": args.gait_bank,
        "cv": {"splits": 5, "repeats": 3, "seed": 20260602},
        "consensus_cv": {
            "ternary_macro_f1": cons_tern_f1,
            "binary_macro_f1": cons_bin_f1,
            "phq_mae": cons_mae,
        },
        "best_single_member_cv": {
            "ternary_macro_f1": best_member_tern,
            "phq_mae": best_member_mae,
        },
        "consensus_beats_best_member": {
            "ternary": cons_tern_f1 >= best_member_tern,
            "phq": cons_mae <= best_member_mae,
        },
        "member_ternary_macro_f1": member_tern_f1,
        "member_phq_mae": member_mae,
        "n_train": int(y_ternary.size),
        "n_test": int(test_table.ids.size),
    }
    (out_dir / "bank_cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "gait_bank": args.gait_bank,
            "clf_members": fitted.clf_members,
            "reg_members": fitted.reg_members,
            "clf_weights": fitted.clf_weights,
            "reg_weights": fitted.reg_weights,
        },
        out_dir / "submission_bank.joblib",
    )

    print(json.dumps(report["consensus_cv"], indent=2))
    print(json.dumps(report["best_single_member_cv"], indent=2))
    print(json.dumps(report["consensus_beats_best_member"], indent=2))
    print(f"[submission] {zip_path}")
    print(f"[bank_cv_report] {out_dir / 'bank_cv_report.json'}")


if __name__ == "__main__":
    main()
