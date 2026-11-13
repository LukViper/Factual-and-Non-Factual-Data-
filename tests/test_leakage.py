"""Leakage validation tests."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

from cross_fold_stacking.exchange import apply_cross_fold_exchange
from cross_fold_stacking.folds import allocate_folds
from cross_fold_stacking.leakage import validate_allocations
from cross_fold_stacking.models import make_base_model
from cross_fold_stacking.oof import generate_base_oof_scores


def test_leakage_ok_on_valid_exchange():
    X, y = make_classification(
        n_samples=400, n_features=20, n_informative=8, n_redundant=2, random_state=0
    )
    folds = allocate_folds(y, n_folds=5, random_seed=0)
    ex = apply_cross_fold_exchange(folds, y, exchange_ratio=0.25, random_seed=0)
    oof = generate_base_oof_scores(X, y, ex, make_base_model("logistic", 0), oof_folds=5, random_seed=0)
    hold_uids = np.arange(1000, 1080)
    dev_uids = np.arange(400)
    report = validate_allocations(ex, 80, hold_uids, dev_uids, oof=oof)
    assert report.ok, report.violations
    assert report.cross_model_train_count > 0  # F2∪F3∪F4


def test_detect_holdout_overlap():
    y = np.array([0, 1] * 100)
    folds = allocate_folds(y, n_folds=5, random_seed=1)
    ex = apply_cross_fold_exchange(folds, y, 0.0, 1)
    bad = validate_allocations(
        ex,
        n_holdout=10,
        holdout_uids=np.array([0, 1, 2]),  # overlap with development uids
        development_uids=np.arange(200),
        oof=None,
    )
    assert not bad.ok
    assert any("Holdout" in v or "overlap" in v for v in bad.violations)
