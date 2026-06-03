"""Build one shard of the baseline model library.

Each candidate model is trained only on the released train labels; we store its
shared-fold out-of-fold predictions (for selection) and its refit-on-full test
predictions. Shardable with --shard / --n-shards for a Slurm array.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from elder_gp_baseline.baseline_model import build_tables, combined_matrix
from elder_gp_baseline.model_library import (
    all_candidates,
    clf_oof_and_test,
    reg_oof_and_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", default="extracted/Train-MPDD-Elder/Elder")
    parser.add_argument("--test-root", default="extracted/Test-MPDD-Elder/Elder")
    parser.add_argument("--label-csv", default="extracted/Train-MPDD-Elder/Elder/split_labels_train.csv")
    parser.add_argument("--personality-npy", default="extracted/Train-MPDD-Elder/Elder/descriptions_embeddings_with_ids.npy")
    parser.add_argument("--library-dir", default="output/library")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()

    lib = Path(args.library_dir).resolve()
    lib.mkdir(parents=True, exist_ok=True)

    train_tables, test_tables = build_tables(
        Path(args.train_root).resolve(),
        Path(args.test_root).resolve(),
        Path(args.label_csv).resolve(),
        Path(args.personality_npy).resolve(),
    )
    any_train = next(iter(train_tables.values()))
    y_ternary = any_train.label3.astype(np.int64)
    y_phq = any_train.phq9.astype(np.float64)
    test_ids = next(iter(test_tables.values())).ids

    # Save shared targets / ids once (shard 0).
    if args.shard == 0:
        np.savez(lib / "_meta.npz", y_ternary=y_ternary, y_phq=y_phq, test_ids=test_ids)

    x_train = {bank: combined_matrix(t) for bank, t in train_tables.items()}
    x_test = {bank: combined_matrix(t) for bank, t in test_tables.items()}

    candidates = all_candidates(tuple(train_tables))
    mine = [c for i, c in enumerate(candidates) if i % args.n_shards == args.shard]
    print(f"[library] shard {args.shard}/{args.n_shards}: {len(mine)} of {len(candidates)} candidates")

    for cand in mine:
        out_path = lib / f"{cand.kind}__{cand.bank}__{cand.name}.npz"
        if out_path.exists():
            continue
        try:
            if cand.kind == "clf":
                oof, test = clf_oof_and_test(cand.make(), x_train[cand.bank], y_ternary, x_test[cand.bank])
            else:
                oof, test = reg_oof_and_test(cand.make(), x_train[cand.bank], y_phq, x_test[cand.bank])
        except Exception as exc:  # a single bad candidate must not kill the shard
            print(f"[library] SKIP {cand.name} ({cand.bank}): {exc}")
            continue
        np.savez(out_path, oof=oof, test=test, kind=cand.kind, bank=cand.bank, name=cand.name)
        print(f"[library] wrote {out_path.name}")

    print(f"[library] shard {args.shard} done.")


if __name__ == "__main__":
    main()
