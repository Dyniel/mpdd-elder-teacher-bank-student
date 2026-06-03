
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    FeatureTable,
    build_test_table,
    build_train_table,
    gait_extractor_for_bank,
)

# PHQ-9 severity band edges that define label3 (and label2 via >= first edge).
PHQ_BAND_EDGES = (5.0, 10.0)
PHQ_MIN, PHQ_MAX = 0.0, 27.0

GAIT_BANKS = ("base", "segment")
PCA_GRID = (None, 16, 32)
LOGREG_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
RIDGE_ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

CV_SPLITS = 5
CV_REPEATS = 5
CV_SEED = 20260602


def phq_to_ternary(phq: np.ndarray) -> np.ndarray:
    """Map a PHQ-9 score to a 3-class severity band."""
    lo, hi = PHQ_BAND_EDGES
    return (np.asarray(phq) >= lo).astype(np.int64) + (np.asarray(phq) >= hi).astype(np.int64)


def ternary_to_binary(ternary: np.ndarray) -> np.ndarray:
    return (np.asarray(ternary) >= 1).astype(np.int64)


def combined_matrix(table: FeatureTable) -> np.ndarray:
    return np.concatenate(
        [table.gait.astype(np.float64), table.personality.astype(np.float64)], axis=1
    )


def _make_clf(pca: int | None, C: float) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if pca is not None:
        steps.append(("pca", PCA(n_components=pca, random_state=CV_SEED)))
    steps.append(
        (
            "clf",
            LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
            ),
        )
    )
    return Pipeline(steps)


def _make_reg(pca: int | None, alpha: float) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if pca is not None:
        steps.append(("pca", PCA(n_components=pca, random_state=CV_SEED)))
    steps.append(("reg", Ridge(alpha=alpha)))
    return Pipeline(steps)


@dataclass
class CandidateResult:
    strategy: str  # "direct_clf" or "phq_binned"
    gait_bank: str
    pca: int | None
    hyperparam: float  # C for direct_clf, ridge alpha for phq_binned
    ternary_macro_f1: float
    binary_macro_f1: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectedModel:
    gait_bank: str
    ternary: CandidateResult
    phq_pca: int | None
    phq_alpha: float
    phq_cv_mae: float
    # Fitted estimators (refit on the full training split).
    ternary_estimator: Any
    phq_estimator: Any
    metadata: dict[str, Any] = field(default_factory=dict)


def _oof_classify(make_est, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Repeated out-of-fold predictions, averaged macro-F1 (ternary, binary).

    Uses StratifiedKFold so the rare classes stay represented in every fold.
    Returns an array [mean_ternary_macro_f1, mean_binary_macro_f1].
    """
    tern_f1, bin_f1 = [], []
    for repeat in range(CV_REPEATS):
        skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED + repeat)
        oof = np.empty_like(y)
        for train_idx, test_idx in skf.split(x, y):
            est = clone(make_est)
            est.fit(x[train_idx], y[train_idx])
            oof[test_idx] = est.predict(x[test_idx])
        tern_f1.append(f1_score(y, oof, average="macro"))
        bin_f1.append(f1_score(ternary_to_binary(y), ternary_to_binary(oof), average="macro"))
    return np.array([float(np.mean(tern_f1)), float(np.mean(bin_f1))])


def _oof_phq(make_est, x: np.ndarray, y_phq: np.ndarray, y_ternary: np.ndarray) -> tuple[float, float, float]:
    """Repeated out-of-fold PHQ regression.

    Returns (mean MAE, mean ternary macro-F1 from binning, mean binary macro-F1
    from binning). KFold (not stratified) is correct for a continuous target.
    """
    maes, tern_f1, bin_f1 = [], [], []
    for repeat in range(CV_REPEATS):
        kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED + repeat)
        oof = np.empty_like(y_phq)
        for train_idx, test_idx in kf.split(x):
            est = clone(make_est)
            est.fit(x[train_idx], y_phq[train_idx])
            oof[test_idx] = np.clip(est.predict(x[test_idx]), PHQ_MIN, PHQ_MAX)
        maes.append(mean_absolute_error(y_phq, oof))
        binned = phq_to_ternary(oof)
        tern_f1.append(f1_score(y_ternary, binned, average="macro"))
        bin_f1.append(f1_score(ternary_to_binary(y_ternary), ternary_to_binary(binned), average="macro"))
    return float(np.mean(maes)), float(np.mean(tern_f1)), float(np.mean(bin_f1))


def evaluate_candidates(
    tables: dict[str, FeatureTable],
    y_ternary: np.ndarray,
    y_phq: np.ndarray,
) -> tuple[list[CandidateResult], dict[str, tuple[int | None, float, float]]]:
    """Cross-validate every (bank, strategy, hyper-parameter) candidate.

    Returns the list of classification candidates and, per gait bank, the best
    PHQ ridge regressor (pca, alpha, cv_mae) selected purely by CV MAE.
    """
    candidates: list[CandidateResult] = []
    phq_best: dict[str, tuple[int | None, float, float]] = {}

    for bank, table in tables.items():
        x = combined_matrix(table)

        # --- PHQ regressor selection (by CV MAE) ---
        best_phq: tuple[int | None, float, float] | None = None
        for pca in PCA_GRID:
            for alpha in RIDGE_ALPHA_GRID:
                mae, _, _ = _oof_phq(_make_reg(pca, alpha), x, y_phq, y_ternary)
                if best_phq is None or mae < best_phq[2]:
                    best_phq = (pca, alpha, mae)
        assert best_phq is not None
        phq_best[bank] = best_phq

        # Strategy A: derive ternary by binning the PHQ regressor selected above.
        pca_b, alpha_b, _ = best_phq
        _, tern_f1, bin_f1 = _oof_phq(_make_reg(pca_b, alpha_b), x, y_phq, y_ternary)
        candidates.append(
            CandidateResult(
                strategy="phq_binned",
                gait_bank=bank,
                pca=pca_b,
                hyperparam=alpha_b,
                ternary_macro_f1=tern_f1,
                binary_macro_f1=bin_f1,
            )
        )

        # Strategy B: direct multinomial logistic regression on ternary labels.
        for pca in PCA_GRID:
            for C in LOGREG_C_GRID:
                tern_f1, bin_f1 = _oof_classify(_make_clf(pca, C), x, y_ternary)
                candidates.append(
                    CandidateResult(
                        strategy="direct_clf",
                        gait_bank=bank,
                        pca=pca,
                        hyperparam=C,
                        ternary_macro_f1=tern_f1,
                        binary_macro_f1=bin_f1,
                    )
                )

    return candidates, phq_best


def select_and_fit(
    tables: dict[str, FeatureTable],
    y_ternary: np.ndarray,
    y_phq: np.ndarray,
) -> tuple[SelectedModel, list[CandidateResult]]:
    candidates, phq_best = evaluate_candidates(tables, y_ternary, y_phq)

    # Primary selection: best held-out ternary macro-F1, tie-break by binary F1.
    best = max(candidates, key=lambda c: (c.ternary_macro_f1, c.binary_macro_f1))
    bank = best.gait_bank
    x = combined_matrix(tables[bank])

    phq_pca, phq_alpha, phq_mae = phq_best[bank]
    phq_estimator = _make_reg(phq_pca, phq_alpha).fit(x, y_phq)

    if best.strategy == "direct_clf":
        ternary_estimator: Any = _make_clf(best.pca, best.hyperparam).fit(x, y_ternary)
    else:
        # phq_binned: ternary is produced from the PHQ regressor at inference.
        ternary_estimator = None

    selected = SelectedModel(
        gait_bank=bank,
        ternary=best,
        phq_pca=phq_pca,
        phq_alpha=phq_alpha,
        phq_cv_mae=phq_mae,
        ternary_estimator=ternary_estimator,
        phq_estimator=phq_estimator,
        metadata={
            "n_train": int(y_ternary.size),
            "n_features": int(x.shape[1]),
            "cv": {"splits": CV_SPLITS, "repeats": CV_REPEATS, "seed": CV_SEED},
            "phq_band_edges": list(PHQ_BAND_EDGES),
        },
    )
    return selected, candidates


def predict_test(model: SelectedModel, test_table: FeatureTable) -> dict[str, np.ndarray]:
    x = combined_matrix(test_table)
    phq = np.clip(model.phq_estimator.predict(x), PHQ_MIN, PHQ_MAX)
    if model.ternary.strategy == "direct_clf":
        ternary = model.ternary_estimator.predict(x).astype(np.int64)
    else:
        ternary = phq_to_ternary(phq)
    binary = ternary_to_binary(ternary)
    return {"ids": test_table.ids, "phq9": phq, "ternary": ternary, "binary": binary}


def build_tables(
    train_root: Path, test_root: Path, label_csv: Path, personality_npy: Path
) -> tuple[dict[str, FeatureTable], dict[str, FeatureTable]]:
    train_tables: dict[str, FeatureTable] = {}
    test_tables: dict[str, FeatureTable] = {}
    for bank in GAIT_BANKS:
        extractor = gait_extractor_for_bank(bank, seed=CV_SEED)
        train_tables[bank] = build_train_table(train_root, label_csv, personality_npy, extractor)
        test_tables[bank] = build_test_table(test_root, personality_npy, extractor)
    return train_tables, test_tables


def candidate_to_dict(c: CandidateResult) -> dict[str, Any]:
    return {
        "strategy": c.strategy,
        "gait_bank": c.gait_bank,
        "pca": c.pca,
        "hyperparam": c.hyperparam,
        "ternary_macro_f1": c.ternary_macro_f1,
        "binary_macro_f1": c.binary_macro_f1,
    }


def cv_report(model: SelectedModel, candidates: list[CandidateResult]) -> dict[str, Any]:
    ranked = sorted(candidates, key=lambda c: (-c.ternary_macro_f1, -c.binary_macro_f1))
    return {
        "selected": {
            "gait_bank": model.gait_bank,
            "ternary_strategy": model.ternary.strategy,
            "ternary_pca": model.ternary.pca,
            "ternary_hyperparam": model.ternary.hyperparam,
            "cv_ternary_macro_f1": model.ternary.ternary_macro_f1,
            "cv_binary_macro_f1": model.ternary.binary_macro_f1,
            "phq_pca": model.phq_pca,
            "phq_alpha": model.phq_alpha,
            "cv_phq_mae": model.phq_cv_mae,
        },
        "metadata": model.metadata,
        "candidate_ranking": [candidate_to_dict(c) for c in ranked],
    }
