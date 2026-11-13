"""Composition reporting under exchange-origin definition."""
from __future__ import annotations

import numpy as np

from cross_fold_stacking.distributions import assign_distribution_labels, report_compositions
from cross_fold_stacking.exchange import apply_cross_fold_exchange
from cross_fold_stacking.folds import allocate_folds
import pandas as pd


def test_exchange_origin_composition_formula():
    y = np.array([0, 1] * 100)
    folds = allocate_folds(y, 5, 0)
    # fold size 40; r=0.25 → |A|=|B|=10; dissimilar_pct = 10/160 = 0.0625
    ex = apply_cross_fold_exchange(folds, y, 0.25, 0)
    dummy = pd.DataFrame({"claim": ["x"] * 200})
    labels = assign_distribution_labels(dummy, "exchange_origin")
    comps = report_compositions(ex, labels, "exchange_origin")
    for side in ("base", "meta"):
        assert comps[side].dissimilar_count == ex.n_exchanged
        assert abs(comps[side].dissimilar_pct - ex.n_exchanged / comps[side].n_train) < 1e-9
        assert abs(comps[side].similar_pct - (1 - comps[side].dissimilar_pct)) < 1e-9
        assert comps[side].target_achieved is False  # 6.25% ≠ 25%
