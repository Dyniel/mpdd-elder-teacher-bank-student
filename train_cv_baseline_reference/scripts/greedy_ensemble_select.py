
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np

from elder_gp_baseline.baseline_model import phq_to_ternary, ternary_to_binary
from elder_gp_baseline.model_library import (
    blend_clf_test,
    blend_reg_test,
    greedy_select_clf,
    greedy_select_reg,
)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_submission(out_dir: Path, ids, phq, binary, ternary) -> Path:
    sub = out_dir / "submission"
    _write_csv(sub / "binary.csv",
               [{"id": int(i), "binary_pred": int(b), "phq9_pred": f"{p:.8f}"} for i, b, p in zip(ids, binary, phq, strict=True)],
               ["id", "binary_pred", "phq9_pred"])
    _write_csv(sub / "ternary.csv",
               [{"id": int(i), "ternary_pred": int(t), "phq9_pred": f"{p:.8f}"} for i, t, p in zip(ids, ternary, phq, strict=True)],
               ["id", "ternary_pred", "phq9_pred"])
    zip_path = sub / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(sub / "binary.csv", arcname="binary.csv")
        archive.write(sub / "ternary.csv", arcname="ternary.csv")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir", default="output/library")
    parser.add_argument("--out-dir", default="output/ensemble")
    parser.add_argument("--ternary-from", choices=("clf", "phq_binned"), default="clf",
                        help="how the final ternary class is produced")
    parser.add_argument("--max-rounds", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    lib = Path(args.library_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = np.load(lib / "_meta.npz")
    y_ternary = meta["y_ternary"].astype(np.int64)
    y_phq = meta["y_phq"].astype(np.float64)
    test_ids = meta["test_ids"]

    clf_oof, clf_test, reg_oof, reg_test = {}, {}, {}, {}
    for path in sorted(lib.glob("*.npz")):
        if path.name == "_meta.npz":
            continue
        d = np.load(path, allow_pickle=False)
        key = f"{str(d['bank'])}/{str(d['name'])}"
        if str(d["kind"]) == "clf":
            clf_oof[key], clf_test[key] = d["oof"], d["test"]
        else:
            reg_oof[key], reg_test[key] = d["oof"], d["test"]

    if not clf_oof or not reg_oof:
        raise SystemExit(f"library incomplete: {len(clf_oof)} clf, {len(reg_oof)} reg candidates")
    print(f"[select] loaded {len(clf_oof)} classifiers, {len(reg_oof)} regressors")

    clf_w, clf_traj, clf_cv_f1 = greedy_select_clf(clf_oof, y_ternary, args.max_rounds, args.patience)
    reg_w, reg_traj, reg_cv_mae, reg_binned_f1 = greedy_select_reg(reg_oof, y_phq, y_ternary, args.max_rounds, args.patience)

    phq = blend_reg_test(reg_w, reg_test)
    if args.ternary_from == "clf":
        ternary = blend_clf_test(clf_w, clf_test)
        final_cv_f1 = clf_cv_f1
    else:
        ternary = phq_to_ternary(phq)
        final_cv_f1 = reg_binned_f1
    binary = ternary_to_binary(ternary)

    zip_path = _write_submission(out_dir, test_ids, phq, binary, ternary)
    _write_csv(out_dir / "selected_test_predictions.csv",
               [{"id": int(i), "binary_pred": int(b), "ternary_pred": int(t), "phq9_pred": float(p)}
                for i, b, t, p in zip(test_ids, binary, ternary, phq, strict=True)],
               ["id", "binary_pred", "ternary_pred", "phq9_pred"])

    report = {
        "method": "baseline model library + greedy ensemble selection (Caruana 2004), train-only CV",
        "ternary_source": args.ternary_from,
        "n_candidates": {"clf": len(clf_oof), "reg": len(reg_oof)},
        "selected_cv": {
            "ternary_macro_f1": final_cv_f1,
            "clf_ensemble_cv_ternary_macro_f1": clf_cv_f1,
            "phq_mae": reg_cv_mae,
            "phq_binned_cv_ternary_macro_f1": reg_binned_f1,
        },
        "clf_weights": clf_w,
        "reg_weights": reg_w,
        "clf_trajectory": clf_traj,
        "reg_trajectory": reg_traj,
    }
    (out_dir / "ensemble_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["selected_cv"], indent=2))
    print("clf members:", {k: round(v, 3) for k, v in clf_w.items()})
    print("reg members:", {k: round(v, 3) for k, v in reg_w.items()})
    print(f"[submission] {zip_path}")
    print(f"[report]     {out_dir / 'ensemble_report.json'}")


if __name__ == "__main__":
    main()
