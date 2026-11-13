"""
Stacking inference pipeline: Base scores → Meta final predictions.

Uses the F1–F5 exchange allocation for data roles, and OOF base scores
for meta training / in-development meta evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluate import MetricSet, compute_metrics
from .exchange import ExchangeResult
from .models import clone_model, make_base_model, make_meta_model, predict_scores
from .oof import OOFBundle, generate_base_oof_scores, meta_feature_matrix


@dataclass
class StackingResult:
    exchange_ratio: float
    base_test_metrics: MetricSet
    meta_test_metrics: MetricSet
    stacking_dev_meta_test: MetricSet  # alias of meta_test (final stacking on meta-test)
    holdout_base_metrics: MetricSet
    holdout_stacking_metrics: MetricSet
    oof: OOFBundle
    meta_model: Any
    counts: dict[str, int]


def run_stacking(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    exchange: ExchangeResult,
    base_model_name: str = "logistic",
    meta_model_name: str = "logistic",
    oof_folds: int = 5,
    random_seed: int = 42,
) -> StackingResult:
    """
    Train base + meta under the exchange allocation; evaluate on meta-test
    and the untouched holdout.
    """
    base_est = make_base_model(base_model_name, random_seed)
    meta_est = make_meta_model(meta_model_name, random_seed)

    oof = generate_base_oof_scores(
        X_dev, y_dev, exchange, base_est, oof_folds=oof_folds, random_seed=random_seed
    )

    # --- Base model accuracy on its test split ---
    bt = exchange.base_test_idx
    y_bt_pred, s_bt = predict_scores(oof.base_fitted, X_dev[bt])
    base_test_metrics = compute_metrics("base_test", y_dev[bt], y_bt_pred, s_bt)

    # --- Meta model: train on OOF base scores for meta_train ---
    mt = exchange.meta_train_idx
    mte = exchange.meta_test_idx
    X_meta_tr = meta_feature_matrix(oof.scores, mt)
    X_meta_te = meta_feature_matrix(oof.scores, mte)
    meta = clone_model(meta_est)
    meta.fit(X_meta_tr, y_dev[mt])

    y_meta_pred, s_meta = predict_scores(meta, X_meta_te)
    meta_test_metrics = compute_metrics("meta_test_stacking", y_dev[mte], y_meta_pred, s_meta)

    # --- Untouched holdout ---
    # Base never saw holdout; meta never saw holdout.
    y_h_base, s_h_base = predict_scores(oof.base_fitted, X_holdout)
    holdout_base = compute_metrics("holdout_base", y_holdout, y_h_base, s_h_base)

    X_meta_h = s_h_base.reshape(-1, 1)
    y_h_stack, s_h_stack = predict_scores(meta, X_meta_h)
    holdout_stack = compute_metrics("holdout_stacking", y_holdout, y_h_stack, s_h_stack)

    counts = {
        "base_train": int(len(exchange.base_train_idx)),
        "base_test": int(len(exchange.base_test_idx)),
        "meta_train": int(len(exchange.meta_train_idx)),
        "meta_test": int(len(exchange.meta_test_idx)),
        "holdout": int(len(y_holdout)),
        "n_exchanged_each": int(exchange.n_exchanged),
    }

    return StackingResult(
        exchange_ratio=exchange.exchange_ratio,
        base_test_metrics=base_test_metrics,
        meta_test_metrics=meta_test_metrics,
        stacking_dev_meta_test=meta_test_metrics,
        holdout_base_metrics=holdout_base,
        holdout_stacking_metrics=holdout_stack,
        oof=oof,
        meta_model=meta,
        counts=counts,
    )
