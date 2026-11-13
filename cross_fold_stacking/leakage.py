"""Strict sample-level leakage validation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .exchange import ExchangeResult, ROLE_TEST, ROLE_TRAIN
from .oof import OOFBundle


@dataclass
class LeakageReport:
    ok: bool
    violations: list[str] = field(default_factory=list)
    cross_model_train_count: int = 0
    cross_model_train_indices: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise AssertionError("Leakage checks failed:\n- " + "\n- ".join(self.violations))


def validate_allocations(
    exchange: ExchangeResult,
    n_holdout: int,
    holdout_uids: np.ndarray | None,
    development_uids: np.ndarray,
    oof: OOFBundle | None = None,
) -> LeakageReport:
    """
    Enforce:
      1. Holdout never in train/tune (checked by construction + UID disjointness)
      2. No sample both train and test for the same model
      3. Meta-train features are OOF (if oof provided)
      4. Final eval samples not used in training (holdout disjoint)
      5. Report cross-model training exposure explicitly
    """
    violations: list[str] = []
    notes: list[str] = []

    # Rule 2: role exclusivity per model
    both_base = np.where(
        (exchange.base_role == ROLE_TRAIN) & (exchange.base_role == ROLE_TEST)
    )[0]
    # The above is logically empty by construction; also check set overlap
    overlap_base = np.intersect1d(exchange.base_train_idx, exchange.base_test_idx)
    overlap_meta = np.intersect1d(exchange.meta_train_idx, exchange.meta_test_idx)
    if len(overlap_base):
        violations.append(f"Base train∩test non-empty: {len(overlap_base)} samples")
    if len(overlap_meta):
        violations.append(f"Meta train∩test non-empty: {len(overlap_meta)} samples")

    # Coverage: every sample assigned
    n = len(exchange.base_role)
    if len(exchange.base_train_idx) + len(exchange.base_test_idx) != n:
        violations.append("Base roles do not partition development")
    if len(exchange.meta_train_idx) + len(exchange.meta_test_idx) != n:
        violations.append("Meta roles do not partition development")

    # Exchange size consistency
    if len(exchange.subset_a) != len(exchange.subset_b):
        violations.append(
            f"|A|={len(exchange.subset_a)} != |B|={len(exchange.subset_b)}"
        )
    if exchange.n_exchanged != len(exchange.subset_a):
        violations.append("n_exchanged mismatch with |A|")

    # Subset A must come from F1; B from F5
    if len(exchange.subset_a) and not np.all(exchange.original_fold[exchange.subset_a] == 1):
        violations.append("Subset A contains non-F1 samples")
    if len(exchange.subset_b) and not np.all(exchange.original_fold[exchange.subset_b] == 5):
        violations.append("Subset B contains non-F5 samples")

    # Holdout disjointness
    if holdout_uids is not None:
        inter = np.intersect1d(development_uids, holdout_uids)
        if len(inter):
            violations.append(f"Holdout UIDs overlap development: {len(inter)}")
    notes.append(f"Holdout size recorded: {n_holdout} (must stay untouched)")

    # Rule 3: OOF meta features
    if oof is not None:
        bad = exchange.meta_train_idx[~oof.is_oof[exchange.meta_train_idx]]
        if len(bad):
            violations.append(f"Meta-train lacks OOF base preds: {len(bad)} samples")
        # Meta-test used for stacking eval should also be OOF for fairness
        bad_te = exchange.meta_test_idx[~oof.is_oof[exchange.meta_test_idx]]
        if len(bad_te):
            violations.append(f"Meta-test lacks OOF base preds: {len(bad_te)} samples")

    # Rule 5: cross-model exposure
    cross = np.where(
        (exchange.base_role == ROLE_TRAIN) & (exchange.meta_role == ROLE_TRAIN)
    )[0]
    notes.append(
        f"Cross-model training exposure: {len(cross)} samples "
        f"(expected ≈ F2∪F3∪F4). OOF required for these."
    )

    return LeakageReport(
        ok=len(violations) == 0,
        violations=violations,
        cross_model_train_count=int(len(cross)),
        cross_model_train_indices=cross.tolist(),
        notes=notes,
    )


def leakage_to_frame(report: LeakageReport) -> pd.DataFrame:
    rows = [{"ok": report.ok, "cross_model_train_count": report.cross_model_train_count}]
    for i, v in enumerate(report.violations):
        rows.append({"ok": False, "violation": v})
    for n in report.notes:
        rows.append({"ok": report.ok, "note": n})
    return pd.DataFrame(rows)
