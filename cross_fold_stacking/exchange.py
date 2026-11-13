"""
Cross-fold sample-level exchange between F1 and F5.

Preserves F2/F3/F4 as shared training for both models.
Does NOT exchange entire folds — only subsets A ⊂ F1 and B ⊂ F5 of equal size.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .folds import FoldAllocation


ROLE_TRAIN = "train"
ROLE_TEST = "test"


@dataclass
class ExchangeResult:
    """Sample-level allocations after optional F1↔F5 exchange."""

    exchange_ratio: float
    n_exchanged: int
    subset_a: np.ndarray  # indices (into development) from F1 moved
    subset_b: np.ndarray  # indices from F5 moved

    # Per-sample role arrays length n_dev
    base_role: np.ndarray  # "train" | "test"
    meta_role: np.ndarray

    exchanged: np.ndarray  # bool
    original_fold: np.ndarray

    # Convenience index sets
    base_train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    base_test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    meta_train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    meta_test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def manifest(self, development: pd.DataFrame) -> pd.DataFrame:
        """Sample allocation manifest for reproducibility / leakage audits."""
        df = development.copy()
        df["original_fold"] = self.original_fold
        df["fold_name"] = [f"F{f}" for f in self.original_fold]
        df["exchanged"] = self.exchanged
        df["base_role"] = self.base_role
        df["meta_role"] = self.meta_role
        df["in_base_train"] = self.base_role == ROLE_TRAIN
        df["in_base_test"] = self.base_role == ROLE_TEST
        df["in_meta_train"] = self.meta_role == ROLE_TRAIN
        df["in_meta_test"] = self.meta_role == ROLE_TEST
        df["cross_model_train_exposure"] = (
            (self.base_role == ROLE_TRAIN) & (self.meta_role == ROLE_TRAIN)
        )
        return df


def _stratified_subset(
    indices: np.ndarray,
    y: np.ndarray,
    n_take: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick n_take indices from `indices`, stratified by y when possible."""
    if n_take <= 0:
        return np.array([], dtype=int)
    if n_take > len(indices):
        raise ValueError(f"Cannot take {n_take} from fold of size {len(indices)}")

    indices = np.asarray(indices, dtype=int)
    labels = y[indices]
    classes, counts = np.unique(labels, return_counts=True)
    # Proportional quota per class
    quotas = {c: int(round(n_take * (cnt / len(indices)))) for c, cnt in zip(classes, counts)}
    # Fix rounding so sum == n_take
    while sum(quotas.values()) > n_take:
        c = max(quotas, key=quotas.get)
        quotas[c] -= 1
    while sum(quotas.values()) < n_take:
        c = max(classes, key=lambda cl: counts[list(classes).index(cl)] - quotas[cl])
        quotas[c] += 1

    chosen = []
    for c in classes:
        pool = indices[labels == c]
        q = quotas[c]
        if q > len(pool):
            q = len(pool)
        pick = rng.choice(pool, size=q, replace=False)
        chosen.append(pick)
    selected = np.concatenate(chosen) if chosen else np.array([], dtype=int)

    # Top-up / trim if stratification edge cases
    if len(selected) < n_take:
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        need = n_take - len(selected)
        selected = np.concatenate(
            [selected, rng.choice(remaining, size=need, replace=False)]
        )
    elif len(selected) > n_take:
        selected = rng.choice(selected, size=n_take, replace=False)
    return np.sort(selected.astype(int))


def apply_cross_fold_exchange(
    folds: FoldAllocation,
    y: np.ndarray,
    exchange_ratio: float,
    random_seed: int = 42,
) -> ExchangeResult:
    """
    Build Base/Meta train/test roles with sample-level F1↔F5 exchange.

    exchange_ratio ∈ [0, 1]: fraction of each of F1 and F5 to swap (|A|=|B|).
    Uses min(|F1|,|F5|) * ratio so counts stay equal when folds differ by 1.
    """
    if not 0.0 <= exchange_ratio <= 1.0:
        raise ValueError("exchange_ratio must be in [0, 1]")

    y = np.asarray(y)
    n = len(y)
    f1 = folds.indices(1)
    f5 = folds.indices(5)
    n_exchange = int(round(exchange_ratio * min(len(f1), len(f5))))

    rng = np.random.default_rng(random_seed + int(round(exchange_ratio * 10_000)))
    subset_a = _stratified_subset(f1, y, n_exchange, rng)
    subset_b = _stratified_subset(f5, y, n_exchange, rng)

    a_set = set(subset_a.tolist())
    b_set = set(subset_b.tolist())

    base_role = np.empty(n, dtype=object)
    meta_role = np.empty(n, dtype=object)
    exchanged = np.zeros(n, dtype=bool)

    for i in range(n):
        fold = int(folds.fold_id[i])
        is_a = i in a_set
        is_b = i in b_set
        exchanged[i] = is_a or is_b

        # Spec §3 table
        if fold == 1 and not is_a:
            base_role[i], meta_role[i] = ROLE_TRAIN, ROLE_TEST
        elif fold == 1 and is_a:
            base_role[i], meta_role[i] = ROLE_TEST, ROLE_TRAIN
        elif fold in (2, 3, 4):
            base_role[i], meta_role[i] = ROLE_TRAIN, ROLE_TRAIN
        elif fold == 5 and not is_b:
            base_role[i], meta_role[i] = ROLE_TEST, ROLE_TRAIN
        elif fold == 5 and is_b:
            base_role[i], meta_role[i] = ROLE_TRAIN, ROLE_TEST
        else:
            raise RuntimeError(f"Unhandled fold={fold} for sample {i}")

    result = ExchangeResult(
        exchange_ratio=exchange_ratio,
        n_exchanged=n_exchange,
        subset_a=subset_a,
        subset_b=subset_b,
        base_role=base_role,
        meta_role=meta_role,
        exchanged=exchanged,
        original_fold=folds.fold_id.copy(),
    )
    result.base_train_idx = np.where(base_role == ROLE_TRAIN)[0]
    result.base_test_idx = np.where(base_role == ROLE_TEST)[0]
    result.meta_train_idx = np.where(meta_role == ROLE_TRAIN)[0]
    result.meta_test_idx = np.where(meta_role == ROLE_TEST)[0]
    return result
