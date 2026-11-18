"""
Cross-fold data-exchange stacking with multi-base OOF features.

Bases = all individual classifiers from the 17-suite.
Meta  = top-performing classifier (validation rank on development).

Preserves F1–F5 allocation + sample exchange from cross_fold_stacking/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from cross_fold_stacking.exchange import apply_cross_fold_exchange
from cross_fold_stacking.folds import allocate_folds

from .classifiers import get_base_pool, make_meta_estimator, positive_scores, wrap_for_proba
from .metrics_util import compute_metrics
from .ranking import rank_base_models, top_meta_names


def _oof_scores_one_base(
    est,
    X: np.ndarray,
    y: np.ndarray,
    base_train_idx: np.ndarray,
    base_test_idx: np.ndarray,
    oof_folds: int,
    random_state: int,
) -> tuple[np.ndarray, object]:
    """OOF scores for one base under exchange roles; return (scores_dev, fitted_full)."""
    n = len(y)
    scores = np.full(n, np.nan, dtype=np.float64)

    full = clone(est)
    full.fit(X[base_train_idx], y[base_train_idx])

    if len(base_test_idx):
        _, s = positive_scores(full, X[base_test_idx])
        scores[base_test_idx] = s

    y_bt = y[base_train_idx]
    _, counts = np.unique(y_bt, return_counts=True)
    n_splits = max(2, min(oof_folds, int(counts.min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_rel, te_rel in skf.split(np.zeros(len(base_train_idx)), y_bt):
        tr_abs = base_train_idx[tr_rel]
        te_abs = base_train_idx[te_rel]
        clf = clone(est)
        clf.fit(X[tr_abs], y[tr_abs])
        _, s = positive_scores(clf, X[te_abs])
        scores[te_abs] = s

    if np.isnan(scores).any():
        raise RuntimeError("Missing OOF scores for a base model")
    return scores, full


def _build_base_oof_matrix(
    X: np.ndarray,
    y: np.ndarray,
    base_train_idx: np.ndarray,
    base_test_idx: np.ndarray,
    random_state: int,
    oof_folds: int = 3,
) -> tuple[np.ndarray, list[str], list]:
    """Shape (n_dev, n_bases) OOF score matrix + fitted full bases."""
    large = len(y) > 10000
    pool = get_base_pool(random_state, large_data=large)
    cols = []
    names = []
    fitted = []
    for name, est in pool.items():
        print(f"    OOF base: {name}", flush=True)
        model = wrap_for_proba(name, est)
        scores, full = _oof_scores_one_base(
            model, X, y, base_train_idx, base_test_idx, oof_folds, random_state
        )
        cols.append(scores.reshape(-1, 1))
        names.append(name)
        fitted.append(full)
    return np.hstack(cols), names, fitted


def run_cross_fold_stacking(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    *,
    size_label: str,
    embedder: str,
    feature_name: str,
    exchange_ratios: list[float] | None = None,
    random_state: int = 42,
    top_k_meta: int = 3,
    oof_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if exchange_ratios is None:
        exchange_ratios = [0.0, 0.10, 0.25, 0.40]
    if oof_folds is None:
        oof_folds = 2 if len(y_dev) > 10000 else 3

    print(f"[cross-fold] Ranking bases on development ({len(y_dev)} rows)...")
    ranking = rank_base_models(X_dev, y_dev, random_state=random_state, val_fraction=0.2)
    metas = top_meta_names(ranking, top_k=top_k_meta)
    print(f"[cross-fold] Top-{top_k_meta} meta candidates: {metas}")

    folds = allocate_folds(y_dev, n_folds=5, random_seed=random_state)
    summary_rows = []
    config_rows = []
    exp_label = {0.0: "E0", 0.10: "E1", 0.25: "E2", 0.40: "E3"}

    for ratio in exchange_ratios:
        label = exp_label.get(float(ratio), f"r{ratio}")
        print(f"[cross-fold] {label} exchange={ratio}")
        exchange = apply_cross_fold_exchange(
            folds, y_dev, exchange_ratio=float(ratio), random_seed=random_state
        )

        oof_mat, base_names, fitted_bases = _build_base_oof_matrix(
            X_dev,
            y_dev,
            exchange.base_train_idx,
            exchange.base_test_idx,
            random_state,
            oof_folds=oof_folds,
        )

        # Holdout features from full bases
        hold_cols = []
        for full in fitted_bases:
            _, s = positive_scores(full, X_holdout)
            hold_cols.append(s.reshape(-1, 1))
        X_hold_meta = np.hstack(hold_cols)

        mt = exchange.meta_train_idx
        mte = exchange.meta_test_idx
        bt = exchange.base_test_idx

        for meta_name in metas:
            meta = make_meta_estimator(
                meta_name, random_state, large_data=len(y_dev) > 10000
            )
            meta.fit(oof_mat[mt], y_dev[mt])

            # Average base scores on base-test as a simple base metric
            y_bt = (oof_mat[bt].mean(axis=1) >= 0.5).astype(int)
            base_m = compute_metrics("base_test_mean_score", y_dev[bt], y_bt, oof_mat[bt].mean(axis=1))

            y_meta, s_meta = positive_scores(meta, oof_mat[mte])
            meta_m = compute_metrics("meta_test_stacking", y_dev[mte], y_meta, s_meta)

            y_h, s_h = positive_scores(meta, X_hold_meta)
            hold_m = compute_metrics("holdout_stacking", y_holdout, y_h, s_h)

            common = {
                "method": "cross_fold_stacking",
                "size": size_label,
                "embedder": embedder,
                "feature_name": feature_name,
                "experiment": label,
                "exchange_ratio": float(ratio),
                "n_exchanged_each": int(exchange.n_exchanged),
                "n_bases": len(base_names),
                "meta_model": meta_name,
                "meta_rank": int(ranking.loc[ranking.classifier == meta_name, "rank"].iloc[0]),
                "meta_val_accuracy": float(
                    ranking.loc[ranking.classifier == meta_name, "val_accuracy"].iloc[0]
                ),
                "base_train": int(len(exchange.base_train_idx)),
                "base_test": int(len(exchange.base_test_idx)),
                "meta_train": int(len(exchange.meta_train_idx)),
                "meta_test": int(len(exchange.meta_test_idx)),
                "holdout": int(len(y_holdout)),
            }
            for m in (base_m, meta_m, hold_m):
                summary_rows.append({**common, **m})

            config_rows.append(
                {
                    **{k: common[k] for k in (
                        "method", "size", "embedder", "experiment", "exchange_ratio",
                        "meta_model", "n_bases",
                    )},
                    "base_names": ", ".join(base_names),
                }
            )
            print(
                f"  meta={meta_name}: holdout_acc={hold_m['accuracy']:.4f} "
                f"f1={hold_m['f1']:.4f}"
            )

    ranking = ranking.assign(size=size_label, embedder=embedder, method="cross_fold_stacking")
    return pd.DataFrame(summary_rows), ranking, pd.DataFrame(config_rows)
