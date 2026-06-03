"""Baseline submission bank: bank -> consensus -> final submission.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .baseline_model import (
    PHQ_MAX,
    PHQ_MIN,
    phq_to_ternary,
    ternary_to_binary,
)

BANK_SPLITS = 5
BANK_REPEATS = 3
BANK_SEED = 20260602


@dataclass(frozen=True)
class Member:
    name: str
    make: Callable[[], Pipeline]


def _scaled(*tail) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), *tail])


def classifier_bank() -> list[Member]:
    """Diverse, individually regularized ternary classifiers."""
    return [
        Member("logreg_C0.1", lambda: _scaled(("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)))),
        Member("logreg_C0.3", lambda: _scaled(("clf", LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)))),
        Member("logreg_C1.0", lambda: _scaled(("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)))),
        Member("logreg_pca32", lambda: _scaled(("pca", PCA(n_components=32, random_state=BANK_SEED)), ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)))),
        Member("svc_linear", lambda: _scaled(("pca", PCA(n_components=32, random_state=BANK_SEED)), ("clf", SVC(kernel="linear", C=0.5, class_weight="balanced")))),
        Member("extratrees", lambda: _scaled(("clf", ExtraTreesClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=BANK_SEED)))),
    ]


def regressor_bank() -> list[Member]:
    """Diverse PHQ-9 regressors."""
    return [
        Member("ridge_a100_pca16", lambda: _scaled(("pca", PCA(n_components=16, random_state=BANK_SEED)), ("reg", Ridge(alpha=100.0)))),
        Member("ridge_a300_pca32", lambda: _scaled(("pca", PCA(n_components=32, random_state=BANK_SEED)), ("reg", Ridge(alpha=300.0)))),
        Member("ridge_a1000", lambda: _scaled(("reg", Ridge(alpha=1000.0)))),
        Member("knn_reg", lambda: _scaled(("pca", PCA(n_components=16, random_state=BANK_SEED)), ("reg", KNeighborsRegressor(n_neighbors=7, weights="distance")))),
        Member("rf_reg", lambda: _scaled(("reg", RandomForestRegressor(n_estimators=400, max_depth=6, random_state=BANK_SEED)))),
    ]


def weighted_vote(member_preds: np.ndarray, weights: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    """Per-sample weighted vote. Ties go to the lower label (matches the original)."""
    n = member_preds.shape[1]
    out = np.empty(n, dtype=np.int64)
    for j in range(n):
        votes: dict[int, float] = defaultdict(float)
        for m in range(member_preds.shape[0]):
            votes[int(member_preds[m, j])] += float(weights[m])
        out[j] = max(labels, key=lambda label: (votes[label], -label))
    return out


def _clf_weights(members: list[Member], weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.ones(len(members), dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)


def oof_ternary_consensus(
    x: np.ndarray, y_ternary: np.ndarray, members: list[Member], weights: np.ndarray | None = None
) -> tuple[float, float, dict[str, float]]:
    """Out-of-fold weighted-vote consensus + each member's own OOF macro-F1."""
    w = _clf_weights(members, weights)
    tern_f1, bin_f1 = [], []
    member_tern_f1: dict[str, list[float]] = {m.name: [] for m in members}
    for repeat in range(BANK_REPEATS):
        skf = StratifiedKFold(n_splits=BANK_SPLITS, shuffle=True, random_state=BANK_SEED + repeat)
        oof = np.empty_like(y_ternary)
        oof_members = {m.name: np.empty_like(y_ternary) for m in members}
        for tr, te in skf.split(x, y_ternary):
            preds = np.empty((len(members), te.size), dtype=np.int64)
            for i, m in enumerate(members):
                p = clone(m.make()).fit(x[tr], y_ternary[tr]).predict(x[te]).astype(np.int64)
                preds[i] = p
                oof_members[m.name][te] = p
            oof[te] = weighted_vote(preds, w, (0, 1, 2))
        tern_f1.append(f1_score(y_ternary, oof, average="macro"))
        bin_f1.append(f1_score(ternary_to_binary(y_ternary), ternary_to_binary(oof), average="macro"))
        for m in members:
            member_tern_f1[m.name].append(f1_score(y_ternary, oof_members[m.name], average="macro"))
    member_means = {name: float(np.mean(v)) for name, v in member_tern_f1.items()}
    return float(np.mean(tern_f1)), float(np.mean(bin_f1)), member_means


def oof_phq_consensus(
    x: np.ndarray, y_phq: np.ndarray, y_ternary: np.ndarray, members: list[Member], weights: np.ndarray | None = None
) -> tuple[float, dict[str, float]]:
    """Out-of-fold weighted-mean PHQ consensus + each member's own OOF MAE."""
    w = _clf_weights(members, weights)
    w = w / w.sum()
    maes = []
    member_mae: dict[str, list[float]] = {m.name: [] for m in members}
    for repeat in range(BANK_REPEATS):
        kf = KFold(n_splits=BANK_SPLITS, shuffle=True, random_state=BANK_SEED + repeat)
        oof = np.zeros_like(y_phq)
        oof_members = {m.name: np.empty_like(y_phq) for m in members}
        for tr, te in kf.split(x):
            acc = np.zeros(te.size, dtype=np.float64)
            for i, m in enumerate(members):
                p = np.clip(clone(m.make()).fit(x[tr], y_phq[tr]).predict(x[te]), PHQ_MIN, PHQ_MAX)
                acc += w[i] * p
                oof_members[m.name][te] = p
            oof[te] = acc
        maes.append(mean_absolute_error(y_phq, oof))
        for m in members:
            member_mae[m.name].append(mean_absolute_error(y_phq, oof_members[m.name]))
    member_means = {name: float(np.mean(v)) for name, v in member_mae.items()}
    return float(np.mean(maes)), member_means


@dataclass
class FittedBank:
    clf_members: list[tuple[str, Any]]
    reg_members: list[tuple[str, Any]]
    clf_weights: np.ndarray
    reg_weights: np.ndarray


def fit_bank(x: np.ndarray, y_ternary: np.ndarray, y_phq: np.ndarray) -> FittedBank:
    clf = classifier_bank()
    reg = regressor_bank()
    return FittedBank(
        clf_members=[(m.name, clone(m.make()).fit(x, y_ternary)) for m in clf],
        reg_members=[(m.name, clone(m.make()).fit(x, y_phq)) for m in reg],
        clf_weights=np.ones(len(clf), dtype=np.float64),
        reg_weights=np.ones(len(reg), dtype=np.float64),
    )


def predict_bank(bank: FittedBank, x: np.ndarray) -> dict[str, np.ndarray]:
    clf_preds = np.stack([est.predict(x).astype(np.int64) for _, est in bank.clf_members])
    ternary = weighted_vote(clf_preds, bank.clf_weights, (0, 1, 2))
    w = bank.reg_weights / bank.reg_weights.sum()
    phq = np.zeros(x.shape[0], dtype=np.float64)
    for wi, (_, est) in zip(w, bank.reg_members, strict=True):
        phq += wi * np.clip(est.predict(x), PHQ_MIN, PHQ_MAX)
    return {"ternary": ternary, "binary": ternary_to_binary(ternary), "phq9": phq}
