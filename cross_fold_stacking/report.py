"""Results visualization and Excel / plot reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metrics_rows(experiment_label: str, exchange_ratio: float, result, compositions, leakage) -> list[dict[str, Any]]:
    rows = []
    for m in (
        result.base_test_metrics,
        result.meta_test_metrics,
        result.holdout_base_metrics,
        result.holdout_stacking_metrics,
    ):
        rows.append(
            {
                "experiment": experiment_label,
                "exchange_ratio": exchange_ratio,
                "metric_set": m.name,
                "n": m.n,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "roc_auc": m.roc_auc,
                "tn": m.confusion[0][0],
                "fp": m.confusion[0][1],
                "fn": m.confusion[1][0],
                "tp": m.confusion[1][1],
                "base_train": result.counts["base_train"],
                "base_test": result.counts["base_test"],
                "meta_train": result.counts["meta_train"],
                "meta_test": result.counts["meta_test"],
                "n_exchanged_each": result.counts["n_exchanged_each"],
                "cross_model_train": leakage.cross_model_train_count,
                "base_similar_pct": compositions["base"].similar_pct,
                "base_dissimilar_pct": compositions["base"].dissimilar_pct,
                "meta_similar_pct": compositions["meta"].similar_pct,
                "meta_dissimilar_pct": compositions["meta"].dissimilar_pct,
                "composition_target_achieved_base": compositions["base"].target_achieved,
                "composition_target_achieved_meta": compositions["meta"].target_achieved,
            }
        )
    return rows


def plot_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Bar charts: holdout stacking accuracy / F1 across exchange ratios."""
    hold = summary_df[summary_df["metric_set"] == "holdout_stacking"].copy()
    if hold.empty:
        return
    hold = hold.sort_values("exchange_ratio")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = hold["exchange_ratio"].astype(str)
    axes[0].bar(x, hold["accuracy"], color="#2c7fb8")
    axes[0].set_title("Holdout stacking accuracy")
    axes[0].set_xlabel("Exchange ratio")
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, hold["f1"], color="#41b6c4")
    axes[1].set_title("Holdout stacking F1")
    axes[1].set_xlabel("Exchange ratio")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], title: str, out_path: Path) -> None:
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_excel(
    path: Path,
    summary: pd.DataFrame,
    compositions: pd.DataFrame,
    manifests: dict[str, pd.DataFrame],
    leakage_frames: dict[str, pd.DataFrame],
    config_dict: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        compositions.to_excel(xl, sheet_name="Compositions", index=False)
        pd.DataFrame([config_dict]).to_excel(xl, sheet_name="Config", index=False)
        for name, mf in manifests.items():
            # Keep manifests compact for Excel
            cols = [
                c
                for c in [
                    "sample_uid",
                    "id",
                    "label",
                    "original_fold",
                    "fold_name",
                    "exchanged",
                    "base_role",
                    "meta_role",
                    "cross_model_train_exposure",
                ]
                if c in mf.columns
            ]
            mf[cols].to_excel(xl, sheet_name=f"Manifest_{name}"[:31], index=False)
        for name, lf in leakage_frames.items():
            lf.to_excel(xl, sheet_name=f"Leakage_{name}"[:31], index=False)
