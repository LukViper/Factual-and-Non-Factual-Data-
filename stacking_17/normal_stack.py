"""
Normal stacking with shared OOF features.

Bases = all individual classifiers from the 17-suite.
Meta  = top-K validation-ranked classifiers.

OOF base scores are computed ONCE, then each meta is fit on those
features (avoids refitting all bases for every meta candidate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .classifiers import (
    get_base_pool,
    make_meta_estimator,
    positive_scores,
    wrap_for_proba,
)
from .metrics_util import compute_metrics
from .ranking import rank_base_models, top_meta_names


def _oof_matrix(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    n_splits: int,
) -> tuple[np.ndarray, list[str], list]:
    """Return (n_samples, n_bases) OOF scores + full fitted bases."""
    large = len(y) > 10000
    pool = get_base_pool(random_state, large_data=large)
    names = list(pool.keys())
    n, b = len(y), len(names)
    oof = np.zeros((n, b), dtype=np.float64)
    fitted = []

    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(n_splits, int(counts.min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for j, (name, est) in enumerate(pool.items()):
        print(f"    OOF base [{j+1}/{b}]: {name}", flush=True)
        model = wrap_for_proba(name, est)
        for tr, te in skf.split(X, y):
            clf = clone(model)
            clf.fit(X[tr], y[tr])
            _, s = positive_scores(clf, X[te])
            oof[te, j] = s
        full = clone(model)
        full.fit(X, y)
        fitted.append(full)
    return oof, names, fitted


def run_normal_stacking(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    *,
    size_label: str,
    embedder: str,
    feature_name: str,
    random_state: int = 42,
    top_k_meta: int = 3,
    stack_cv: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on development; evaluate on untouched holdout."""
    if stack_cv is None:
        stack_cv = 2 if len(y_dev) > 10000 else 3

    print(f"[normal] Ranking bases on development ({len(y_dev)} rows)...")
    ranking = rank_base_models(X_dev, y_dev, random_state=random_state, val_fraction=0.2)
    metas = top_meta_names(ranking, top_k=top_k_meta)
    print(f"[normal] Top-{top_k_meta} meta candidates: {metas}")

    print(f"[normal] Building shared OOF matrix (cv={stack_cv})...")
    oof, base_names, fitted_bases = _oof_matrix(
        X_dev, y_dev, random_state, n_splits=stack_cv
    )

    hold_cols = []
    for full in fitted_bases:
        _, s = positive_scores(full, X_holdout)
        hold_cols.append(s.reshape(-1, 1))
    X_hold_meta = np.hstack(hold_cols)

    summary_rows = []
    config_rows = []

    for meta_name in metas:
        print(f"[normal] Meta={meta_name} on shared OOF features...")
        meta = make_meta_estimator(
            meta_name, random_state, large_data=len(y_dev) > 10000
        )
        meta.fit(oof, y_dev)

        y_tr_hat, s_tr = positive_scores(meta, oof)
        y_te_hat, s_te = positive_scores(meta, X_hold_meta)
        m_tr = compute_metrics("train_dev", y_dev, y_tr_hat, s_tr)
        m_te = compute_metrics("holdout", y_holdout, y_te_hat, s_te)

        for m in (m_tr, m_te):
            summary_rows.append(
                {
                    "method": "normal_stacking",
                    "size": size_label,
                    "embedder": embedder,
                    "feature_name": feature_name,
                    "n_bases": len(base_names),
                    "meta_model": meta_name,
                    "meta_rank": int(
                        ranking.loc[ranking.classifier == meta_name, "rank"].iloc[0]
                    ),
                    "meta_val_accuracy": float(
                        ranking.loc[ranking.classifier == meta_name, "val_accuracy"].iloc[0]
                    ),
                    "stack_cv": stack_cv,
                    **m,
                }
            )
        config_rows.append(
            {
                "method": "normal_stacking",
                "size": size_label,
                "embedder": embedder,
                "meta_model": meta_name,
                "n_dev": len(y_dev),
                "n_holdout": len(y_holdout),
                "base_names": ", ".join(base_names),
            }
        )
        print(
            f"  meta={meta_name}: holdout_acc={m_te['accuracy']:.4f} "
            f"f1={m_te['f1']:.4f} auc={m_te['roc_auc']}"
        )

    ranking = ranking.assign(size=size_label, embedder=embedder, method="normal_stacking")
    return pd.DataFrame(summary_rows), ranking, pd.DataFrame(config_rows)
