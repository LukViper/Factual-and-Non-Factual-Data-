"""Tests for stratified five-fold allocation."""
from __future__ import annotations

import numpy as np

from cross_fold_stacking.folds import allocate_folds, initial_role_table


def test_five_equal_folds_stratified():
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100)
    rng.shuffle(y)
    folds = allocate_folds(y, n_folds=5, random_seed=42)
    sizes = [len(folds.indices(f)) for f in range(1, 6)]
    assert all(s == 40 for s in sizes)
    # Stratification: each fold ~50/50
    for f in range(1, 6):
        idx = folds.indices(f)
        assert abs((y[idx] == 0).sum() - (y[idx] == 1).sum()) <= 2
    # Partition
    all_idx = np.concatenate([folds.indices(f) for f in range(1, 6)])
    assert len(np.unique(all_idx)) == 200


def test_reproducible_fold_assignment():
    y = np.array([0, 1] * 50)
    a = allocate_folds(y, n_folds=5, random_seed=7)
    b = allocate_folds(y, n_folds=5, random_seed=7)
    assert np.array_equal(a.fold_id, b.fold_id)


def test_initial_role_table():
    t = initial_role_table()
    assert list(t["Fold"]) == ["F1", "F2", "F3", "F4", "F5"]
    assert t.loc[t["Fold"] == "F1", "Base Model"].iloc[0] == "Training"
    assert t.loc[t["Fold"] == "F1", "Meta Model"].iloc[0] == "Testing"
    assert t.loc[t["Fold"] == "F5", "Base Model"].iloc[0] == "Testing"
