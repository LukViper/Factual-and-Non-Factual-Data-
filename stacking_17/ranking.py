"""Rank individual classifiers on a validation slice; pick top-K metas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .classifiers import get_base_pool, positive_scores, wrap_for_proba


def rank_base_models(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    val_fraction: float = 0.2,
) -> pd.DataFrame:
    """
    Fit each base on a train slice; score on a held-out validation slice.
    Ranking uses validation accuracy (higher = better meta candidate).
    """
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=val_fraction, random_state=random_state, stratify=y
    )
    rows = []
    large = len(y) > 10000
    pool = get_base_pool(random_state, large_data=large)
    for name, est in pool.items():
        model = wrap_for_proba(name, est)
        try:
            model.fit(X_tr, y_tr)
            y_hat, scores = positive_scores(model, X_va)
            acc = float(accuracy_score(y_va, y_hat))
            ok = True
            err = ""
        except Exception as exc:
            acc = -1.0
            ok = False
            err = f"{exc.__class__.__name__}: {exc}"
        rows.append(
            {
                "classifier": name,
                "val_accuracy": acc,
                "fit_ok": ok,
                "error": err,
                "n_train": len(y_tr),
                "n_val": len(y_va),
            }
        )
        print(f"  rank {name:24s}  val_acc={acc:.4f}" if ok else f"  rank {name}: FAIL {err}")

    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def top_meta_names(ranking: pd.DataFrame, top_k: int = 3) -> list[str]:
    ok = ranking[ranking["fit_ok"]].copy()
    if ok.empty:
        raise RuntimeError("No base model ranked successfully")
    return ok.head(top_k)["classifier"].tolist()
