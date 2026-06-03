"""Baseline model library + greedy ensemble selection (Caruana et al., 2004).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .baseline_model import PHQ_MAX, PHQ_MIN, phq_to_ternary, ternary_to_binary

TERNARY_CLASSES = (0, 1, 2)
FOLD_SPLITS = 5
FOLD_SEED = 20260602


@dataclass(frozen=True)
class Candidate:
    name: str
    bank: str  # "base" or "segment"
    kind: str  # "clf" or "reg"
    make: Callable[[], Any]


def _pipe(*tail) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), *tail])


def _pca(n: int | None):
    return [] if n is None else [("pca", PCA(n_components=n, random_state=FOLD_SEED))]


def classifier_candidates(bank: str) -> list[Candidate]:
    out: list[Candidate] = []
    for pca in (None, 32, 64):
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            out.append(Candidate(f"logreg_C{C}_pca{pca}", bank, "clf",
                                 lambda pca=pca, C=C: _pipe(*_pca(pca), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000)))))
    for C in (0.5, 1.0, 2.0):
        for kernel in ("linear", "rbf"):
            out.append(Candidate(f"svc_{kernel}_C{C}", bank, "clf",
                                 lambda C=C, kernel=kernel: _pipe(*_pca(32), ("clf", SVC(kernel=kernel, C=C, class_weight="balanced", probability=True, random_state=FOLD_SEED)))))
    for depth in (4, 6, 8):
        for seed in (0, 1, 2):
            out.append(Candidate(f"extratrees_d{depth}_s{seed}", bank, "clf",
                                 lambda depth=depth, seed=seed: _pipe(("clf", ExtraTreesClassifier(n_estimators=400, max_depth=depth, class_weight="balanced", random_state=seed)))))
            out.append(Candidate(f"rf_d{depth}_s{seed}", bank, "clf",
                                 lambda depth=depth, seed=seed: _pipe(("clf", RandomForestClassifier(n_estimators=400, max_depth=depth, class_weight="balanced", random_state=seed)))))
    for depth in (2, 3):
        out.append(Candidate(f"gb_clf_d{depth}", bank, "clf",
                             lambda depth=depth: _pipe(("clf", GradientBoostingClassifier(max_depth=depth, n_estimators=200, learning_rate=0.05, random_state=FOLD_SEED)))))
    for k in (5, 7, 9):
        out.append(Candidate(f"knn_clf_k{k}", bank, "clf",
                             lambda k=k: _pipe(*_pca(32), ("clf", KNeighborsClassifier(n_neighbors=k, weights="distance")))))
    return out


def regressor_candidates(bank: str) -> list[Candidate]:
    out: list[Candidate] = []
    for pca in (None, 16, 32, 64):
        for alpha in (30.0, 100.0, 300.0, 1000.0):
            out.append(Candidate(f"ridge_a{alpha}_pca{pca}", bank, "reg",
                                 lambda pca=pca, alpha=alpha: _pipe(*_pca(pca), ("reg", Ridge(alpha=alpha)))))
    for k in (5, 7, 9):
        out.append(Candidate(f"knn_reg_k{k}", bank, "reg",
                             lambda k=k: _pipe(*_pca(16), ("reg", KNeighborsRegressor(n_neighbors=k, weights="distance")))))
    for depth in (4, 6, 8):
        for seed in (0, 1, 2):
            out.append(Candidate(f"rf_reg_d{depth}_s{seed}", bank, "reg",
                                 lambda depth=depth, seed=seed: _pipe(("reg", RandomForestRegressor(n_estimators=400, max_depth=depth, random_state=seed)))))
    for depth in (2, 3):
        out.append(Candidate(f"gb_reg_d{depth}", bank, "reg",
                             lambda depth=depth: _pipe(("reg", GradientBoostingRegressor(max_depth=depth, n_estimators=300, learning_rate=0.05, random_state=FOLD_SEED)))))
    return out


def all_candidates(banks: tuple[str, ...] = ("base", "segment")) -> list[Candidate]:
    out: list[Candidate] = []
    for bank in banks:
        out.extend(classifier_candidates(bank))
        out.extend(regressor_candidates(bank))
    return out


def _align_proba(estimator, proba: np.ndarray) -> np.ndarray:
    """Map predict_proba columns onto fixed [0,1,2], filling missing with 0."""
    classes = list(getattr(estimator, "classes_", TERNARY_CLASSES))
    aligned = np.zeros((proba.shape[0], len(TERNARY_CLASSES)), dtype=np.float64)
    for col, cls in enumerate(classes):
        aligned[:, int(cls)] = proba[:, col]
    return aligned


def clf_oof_and_test(make, x_train: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shared-fold OOF class probabilities (n,3) + refit-on-full test probabilities."""
    skf = StratifiedKFold(n_splits=FOLD_SPLITS, shuffle=True, random_state=FOLD_SEED)
    oof = np.zeros((x_train.shape[0], 3), dtype=np.float64)
    for tr, te in skf.split(x_train, y):
        est = clone(make).fit(x_train[tr], y[tr])
        oof[te] = _align_proba(est, est.predict_proba(x_train[te]))
    full = clone(make).fit(x_train, y)
    test = _align_proba(full, full.predict_proba(x_test))
    return oof, test


def reg_oof_and_test(make, x_train: np.ndarray, y_phq: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=FOLD_SPLITS, shuffle=True, random_state=FOLD_SEED)
    oof = np.zeros(x_train.shape[0], dtype=np.float64)
    for tr, te in kf.split(x_train):
        est = clone(make).fit(x_train[tr], y_phq[tr])
        oof[te] = np.clip(est.predict(x_train[te]), PHQ_MIN, PHQ_MAX)
    full = clone(make).fit(x_train, y_phq)
    test = np.clip(full.predict(x_test), PHQ_MIN, PHQ_MAX)
    return oof, test


# --- greedy ensemble selection -------------------------------------------------

def greedy_select_clf(
    oof: dict[str, np.ndarray], y: np.ndarray, max_rounds: int = 60, patience: int = 15
) -> tuple[dict[str, float], list[dict[str, Any]], float]:
    """Greedy ensemble (with replacement) maximizing OOF ternary macro-F1."""
    names = list(oof)
    init = max(names, key=lambda nm: f1_score(y, oof[nm].argmax(1), average="macro"))
    selected = [init]
    running = oof[init].copy()
    best_f1 = f1_score(y, running.argmax(1), average="macro")
    trajectory = [{"round": 1, "added": init, "cv_ternary_macro_f1": float(best_f1)}]
    best_counts, best_round, since_improve = Counter(selected), 1, 0

    for r in range(2, max_rounds + 1):
        scored = [
            (f1_score(y, ((running + oof[nm]) / r).argmax(1), average="macro"), nm)
            for nm in names
        ]
        score, nm = max(scored, key=lambda s: s[0])
        running += oof[nm]
        selected.append(nm)
        trajectory.append({"round": r, "added": nm, "cv_ternary_macro_f1": float(score)})
        if score > best_f1 + 1e-9:
            best_f1, best_counts, best_round, since_improve = score, Counter(selected), r, 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    total = sum(best_counts.values())
    weights = {nm: cnt / total for nm, cnt in best_counts.items()}
    return weights, trajectory, float(best_f1)


def greedy_select_reg(
    oof: dict[str, np.ndarray], y_phq: np.ndarray, y_ternary: np.ndarray, max_rounds: int = 60, patience: int = 15
) -> tuple[dict[str, float], list[dict[str, Any]], float, float]:
    """Greedy ensemble (with replacement) minimizing OOF PHQ MAE."""
    names = list(oof)
    init = min(names, key=lambda nm: mean_absolute_error(y_phq, oof[nm]))
    selected = [init]
    running = oof[init].copy()
    best_mae = mean_absolute_error(y_phq, running)
    trajectory = [{"round": 1, "added": init, "cv_phq_mae": float(best_mae)}]
    best_counts, since_improve = Counter(selected), 0

    for r in range(2, max_rounds + 1):
        scored = [(mean_absolute_error(y_phq, (running + oof[nm]) / r), nm) for nm in names]
        mae, nm = min(scored, key=lambda s: s[0])
        running += oof[nm]
        selected.append(nm)
        trajectory.append({"round": r, "added": nm, "cv_phq_mae": float(mae)})
        if mae < best_mae - 1e-9:
            best_mae, best_counts, since_improve = mae, Counter(selected), 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    total = sum(best_counts.values())
    weights = {nm: cnt / total for nm, cnt in best_counts.items()}
    # PHQ-binned ternary F1 of the selected blend, as an baseline cross-check.
    blend = np.zeros_like(y_phq)
    for nm, w in weights.items():
        blend += w * oof[nm]
    binned_f1 = f1_score(y_ternary, phq_to_ternary(blend), average="macro")
    return weights, trajectory, float(best_mae), float(binned_f1)


def blend_clf_test(weights: dict[str, float], test_proba: dict[str, np.ndarray]) -> np.ndarray:
    acc = None
    for nm, w in weights.items():
        acc = w * test_proba[nm] if acc is None else acc + w * test_proba[nm]
    ternary = acc.argmax(1).astype(np.int64)
    return ternary


def blend_reg_test(weights: dict[str, float], test_pred: dict[str, np.ndarray]) -> np.ndarray:
    acc = np.zeros(next(iter(test_pred.values())).shape[0], dtype=np.float64)
    for nm, w in weights.items():
        acc += w * test_pred[nm]
    return np.clip(acc, PHQ_MIN, PHQ_MAX)
