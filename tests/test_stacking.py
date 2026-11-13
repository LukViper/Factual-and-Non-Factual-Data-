"""Stacking OOF + end-to-end smoke tests."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

from cross_fold_stacking.exchange import apply_cross_fold_exchange
from cross_fold_stacking.folds import allocate_folds
from cross_fold_stacking.models import make_base_model
from cross_fold_stacking.oof import generate_base_oof_scores
from cross_fold_stacking.stacking import run_stacking


def test_oof_never_uses_in_sample_base_for_meta_train():
    X, y = make_classification(
        n_samples=500, n_features=16, n_informative=6, random_state=42
    )
    folds = allocate_folds(y, n_folds=5, random_seed=42)
    ex = apply_cross_fold_exchange(folds, y, exchange_ratio=0.25, random_seed=42)
    oof = generate_base_oof_scores(
        X, y, ex, make_base_model("logistic", 42), oof_folds=5, random_seed=42
    )
    assert oof.is_oof[ex.meta_train_idx].all()
    assert oof.is_oof[ex.meta_test_idx].all()
    assert not np.isnan(oof.scores).any()


def test_stacking_runs_all_ratios_and_holdout():
    X, y = make_classification(
        n_samples=600,
        n_features=20,
        n_informative=8,
        n_redundant=2,
        weights=[0.5, 0.5],
        random_state=7,
    )
    # Simulate 80/20
    n_hold = 120
    X_dev, y_dev = X[:-n_hold], y[:-n_hold]
    X_h, y_h = X[-n_hold:], y[-n_hold:]
    folds = allocate_folds(y_dev, n_folds=5, random_seed=7)
    for ratio in (0.0, 0.1, 0.25, 0.4):
        ex = apply_cross_fold_exchange(folds, y_dev, ratio, random_seed=7)
        result = run_stacking(
            X_dev, y_dev, X_h, y_h, ex, random_seed=7
        )
        assert 0.0 <= result.holdout_stacking_metrics.accuracy <= 1.0
        assert result.counts["holdout"] == n_hold
        assert result.oof.is_oof[ex.meta_train_idx].all()


def test_exchanged_meta_train_from_f1_uses_base_that_did_not_train_on_them():
    """
    Subset A moves into meta_train and base_test, so full base (fit on base_train)
    never saw A — those scores must be marked OOF.
    """
    X, y = make_classification(n_samples=400, n_features=12, random_state=3)
    folds = allocate_folds(y, 5, 3)
    ex = apply_cross_fold_exchange(folds, y, 0.4, 3)
    assert len(ex.subset_a) > 0
    oof = generate_base_oof_scores(X, y, ex, make_base_model("logistic", 3), 5, 3)
    assert oof.is_oof[ex.subset_a].all()
    # A should be in base_test
    assert set(ex.subset_a).issubset(set(ex.base_test_idx))
