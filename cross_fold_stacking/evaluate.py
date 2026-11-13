"""Metrics, confusion matrices, and structured result records."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricSet:
    name: str
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None) -> MetricSet:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    auc = None
    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = None
    return MetricSet(
        name=name,
        n=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        confusion=cm,
    )
