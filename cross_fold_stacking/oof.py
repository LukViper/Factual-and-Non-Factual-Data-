"""
Out-of-fold base-model prediction generation.

Conflict note
-------------
Under the F1–F5 exchange allocation, F2∪F3∪F4 are training data for *both*
Base and Meta. Conventional stacking forbids feeding in-sample base
predictions to the meta-model. We therefore:

  1. Keep the proposed exchange allocation exactly as specified.
  2. For every meta-training (and meta-test) sample that also sits in
     base_train, generate base scores via inner stratified CV on base_train
     so the predicting base clone never saw that sample.
  3. For samples in base_test (A ∪ F5\\B), use the full base model fitted on
     base_train — those predictions are already out-of-sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .exchange import ExchangeResult
from .models import clone_model, predict_scores


@dataclass
class OOFBundle:
    """Base scores aligned to development index order."""

    scores: np.ndarray  # shape (n_dev,), positive-class score / probability
    is_oof: np.ndarray  # True where score came from a model that never saw the sample
    base_fitted: object  # base model trained on full base_train


def generate_base_oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    exchange: ExchangeResult,
    base_estimator,
    oof_folds: int = 5,
    random_seed: int = 42,
) -> OOFBundle:
    """
    Produce one base score per development sample for meta features / eval.

    Rules:
      - If sample ∉ base_train → score from full base (seen=False) → is_oof=True
      - If sample ∈ base_train → score from CV fold that held it out → is_oof=True
    """
    n = len(y)
    scores = np.full(n, np.nan, dtype=np.float64)
    is_oof = np.zeros(n, dtype=bool)

    base_train = exchange.base_train_idx
    base_test = exchange.base_test_idx

    # Full base on base_train
    base_full = clone_model(base_estimator)
    base_full.fit(X[base_train], y[base_train])

    # Predictions for base_test samples (never in base_train)
    if len(base_test):
        _, s = predict_scores(base_full, X[base_test])
        scores[base_test] = s
        is_oof[base_test] = True

    # Inner CV over base_train for in-train samples
    if len(base_train) < oof_folds * 2:
        # Tiny data fallback: leave-one-out style batched by unique labels
        oof_folds = max(2, min(oof_folds, len(base_train) // 2))

    y_bt = y[base_train]
    # Ensure enough samples per class for stratification
    _, counts = np.unique(y_bt, return_counts=True)
    n_splits = min(oof_folds, int(counts.min()))
    n_splits = max(2, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for tr_rel, te_rel in skf.split(np.zeros(len(base_train)), y_bt):
        tr_abs = base_train[tr_rel]
        te_abs = base_train[te_rel]
        clf = clone_model(base_estimator)
        clf.fit(X[tr_abs], y[tr_abs])
        _, s = predict_scores(clf, X[te_abs])
        scores[te_abs] = s
        is_oof[te_abs] = True

    if np.isnan(scores).any():
        missing = np.where(np.isnan(scores))[0]
        raise RuntimeError(f"Missing OOF scores for indices: {missing[:20]}...")

    return OOFBundle(scores=scores, is_oof=is_oof, base_fitted=base_full)


def meta_feature_matrix(scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Stacking features: currently the base positive-class score (column vector)."""
    return scores[idx].reshape(-1, 1)


def assert_meta_features_are_oof(
    exchange: ExchangeResult,
    is_oof: np.ndarray,
) -> None:
    """Every meta-train sample must have an OOF base prediction."""
    bad = exchange.meta_train_idx[~is_oof[exchange.meta_train_idx]]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} meta-train samples lack OOF base predictions "
            f"(first ids: {bad[:10]})"
        )
