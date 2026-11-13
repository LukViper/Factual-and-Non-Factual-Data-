"""Five-fold stratified allocation of the development set."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


FOLD_NAMES = ("F1", "F2", "F3", "F4", "F5")


@dataclass
class FoldAllocation:
    """Maps each development sample to F1..F5 (1-indexed fold id)."""

    fold_id: np.ndarray  # shape (n_dev,), values in {1,2,3,4,5}
    fold_indices: dict[int, np.ndarray]  # fold -> absolute indices into development

    def indices(self, fold: int) -> np.ndarray:
        return self.fold_indices[fold]

    def as_frame(self, development: pd.DataFrame) -> pd.DataFrame:
        out = development.copy()
        out["original_fold"] = self.fold_id
        out["fold_name"] = [f"F{f}" for f in self.fold_id]
        return out


def allocate_folds(
    y: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> FoldAllocation:
    """
    Stratified equal folds over the development set.

    Fold numbering: first StratifiedKFold test block -> F1, ..., fifth -> F5.
    With n_folds=5 each fold is ~20% of development (as specified).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    y = np.asarray(y)
    n = len(y)
    fold_id = np.zeros(n, dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    for i, (_, test_idx) in enumerate(skf.split(np.zeros(n), y), start=1):
        fold_id[test_idx] = i

    fold_indices = {f: np.where(fold_id == f)[0] for f in range(1, n_folds + 1)}
    return FoldAllocation(fold_id=fold_id, fold_indices=fold_indices)


def initial_role_table() -> pd.DataFrame:
    """Document the pre-exchange Base/Meta roles (spec §2)."""
    rows = [
        ("F1", "Training", "Testing"),
        ("F2", "Training", "Training"),
        ("F3", "Training", "Training"),
        ("F4", "Training", "Training"),
        ("F5", "Testing", "Training"),
    ]
    return pd.DataFrame(rows, columns=["Fold", "Base Model", "Meta Model"])
