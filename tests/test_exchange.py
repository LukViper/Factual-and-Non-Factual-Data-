"""Tests for sample-level F1↔F5 cross-fold exchange."""
from __future__ import annotations

import numpy as np

from cross_fold_stacking.exchange import ROLE_TEST, ROLE_TRAIN, apply_cross_fold_exchange
from cross_fold_stacking.folds import allocate_folds


def _toy_y(n=200):
    return np.array([0] * (n // 2) + [1] * (n // 2))


def test_zero_exchange_matches_initial_allocation():
    y = _toy_y()
    folds = allocate_folds(y, n_folds=5, random_seed=0)
    ex = apply_cross_fold_exchange(folds, y, exchange_ratio=0.0, random_seed=0)
    assert ex.n_exchanged == 0
    f1, f2, f3, f4, f5 = [folds.indices(i) for i in range(1, 6)]
    # Base train = F1+F2+F3+F4; Base test = F5
    assert set(ex.base_train_idx) == set(np.concatenate([f1, f2, f3, f4]))
    assert set(ex.base_test_idx) == set(f5)
    # Meta train = F2+F3+F4+F5; Meta test = F1
    assert set(ex.meta_train_idx) == set(np.concatenate([f2, f3, f4, f5]))
    assert set(ex.meta_test_idx) == set(f1)


def test_exchange_moves_equal_subsets_only():
    y = _toy_y()
    folds = allocate_folds(y, n_folds=5, random_seed=1)
    ex = apply_cross_fold_exchange(folds, y, exchange_ratio=0.25, random_seed=1)
    assert len(ex.subset_a) == len(ex.subset_b) == ex.n_exchanged
    assert ex.n_exchanged > 0
    assert np.all(ex.original_fold[ex.subset_a] == 1)
    assert np.all(ex.original_fold[ex.subset_b] == 5)
    # Sample-level: not entire folds
    assert len(ex.subset_a) < len(folds.indices(1))
    assert len(ex.subset_b) < len(folds.indices(5))


def test_post_exchange_roles_match_spec_table():
    y = _toy_y()
    folds = allocate_folds(y, n_folds=5, random_seed=2)
    ex = apply_cross_fold_exchange(folds, y, exchange_ratio=0.4, random_seed=2)
    a, b = set(ex.subset_a.tolist()), set(ex.subset_b.tolist())
    for i in range(len(y)):
        fold = int(ex.original_fold[i])
        if fold == 1 and i not in a:
            assert ex.base_role[i] == ROLE_TRAIN and ex.meta_role[i] == ROLE_TEST
        elif fold == 1 and i in a:
            assert ex.base_role[i] == ROLE_TEST and ex.meta_role[i] == ROLE_TRAIN
        elif fold in (2, 3, 4):
            assert ex.base_role[i] == ROLE_TRAIN and ex.meta_role[i] == ROLE_TRAIN
        elif fold == 5 and i not in b:
            assert ex.base_role[i] == ROLE_TEST and ex.meta_role[i] == ROLE_TRAIN
        elif fold == 5 and i in b:
            assert ex.base_role[i] == ROLE_TRAIN and ex.meta_role[i] == ROLE_TEST


def test_no_same_model_train_and_test():
    y = _toy_y()
    folds = allocate_folds(y, n_folds=5, random_seed=3)
    for r in (0.0, 0.1, 0.25, 0.4):
        ex = apply_cross_fold_exchange(folds, y, exchange_ratio=r, random_seed=3)
        assert len(np.intersect1d(ex.base_train_idx, ex.base_test_idx)) == 0
        assert len(np.intersect1d(ex.meta_train_idx, ex.meta_test_idx)) == 0
