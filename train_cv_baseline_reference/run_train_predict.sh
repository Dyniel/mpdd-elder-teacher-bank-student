#!/bin/bash
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "$(pwd)/extracted" ]]; then
  ROOT="$(pwd)"
elif [[ -d "$PKG_ROOT/../../extracted" ]]; then
  ROOT="$(cd "$PKG_ROOT/../.." && pwd)"
else
  ROOT="${MPDD_ROOT:-$(pwd)}"
fi

cd "$ROOT"
export PYTHONPATH="$PKG_ROOT/src:${PYTHONPATH:-}"

python "$PKG_ROOT/scripts/train_and_predict.py" \
  --train-root "$ROOT/extracted/Train-MPDD-Elder/Elder" \
  --test-root "$ROOT/extracted/Test-MPDD-Elder/Elder" \
  --label-csv "$ROOT/extracted/Train-MPDD-Elder/Elder/split_labels_train.csv" \
  --personality-npy "$ROOT/extracted/Train-MPDD-Elder/Elder/descriptions_embeddings_with_ids.npy" \
  --out-dir "$PKG_ROOT/output"

echo "Wrote $PKG_ROOT/output/submission/submission.zip"
