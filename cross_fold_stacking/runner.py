"""
Experiment runner: E0–E3 exchange ratios on the cross-fold stacking design.

Preserves existing fever_easy / fever_lib pipelines; this is a separate path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import EXPERIMENT_LABELS, ExperimentConfig
from .data import prepare_dataset
from .distributions import (
    assign_distribution_labels,
    composition_to_frame,
    report_compositions,
)
from .exchange import apply_cross_fold_exchange
from .folds import allocate_folds, initial_role_table
from .leakage import leakage_to_frame, validate_allocations
from .report import metrics_rows, plot_comparison, plot_confusion, write_excel
from .stacking import run_stacking


def run_experiments(cfg: ExperimentConfig) -> Path:
    out = Path(cfg.output_dir) / cfg.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")

    print("=" * 72)
    print("Cross-fold data-exchange stacking")
    print("=" * 72)
    print(json.dumps(cfg.to_dict(), indent=2))

    data = prepare_dataset(
        data_csv=cfg.data_csv,
        sample_total=cfg.sample_total,
        holdout_fraction=cfg.holdout_fraction,
        random_seed=cfg.random_seed,
        embedder=cfg.embedder,
        svd_dim=cfg.svd_dim,
    )
    y_dev = data.development["label"].to_numpy()
    y_hold = data.holdout["label"].to_numpy()
    print(
        f"Development={len(data.development)}  Holdout={len(data.holdout)}  "
        f"Features={data.feature_name} {data.X_dev.shape}"
    )

    folds = allocate_folds(y_dev, n_folds=cfg.n_folds, random_seed=cfg.random_seed)
    for f in range(1, cfg.n_folds + 1):
        print(f"  Fold F{f}: {len(folds.indices(f))} samples")
    initial_role_table().to_csv(out / "initial_role_table.csv", index=False)

    dist_labels = assign_distribution_labels(
        data.development, key=cfg.distribution_key
    )
    data.development.assign(dist_label=dist_labels).to_csv(
        out / "development_with_dist.csv", index=False
    )

    summary_rows: list[dict] = []
    composition_frames: list[pd.DataFrame] = []
    manifests: dict[str, pd.DataFrame] = {}
    leakage_frames: dict[str, pd.DataFrame] = {}

    ratios = cfg.exchange_ratios or [cfg.exchange_ratio]
    for ratio in ratios:
        label = EXPERIMENT_LABELS.get(float(ratio), f"E_r{ratio}")
        print("-" * 72)
        print(f"Experiment {label}: exchange_ratio={ratio}")

        exchange = apply_cross_fold_exchange(
            folds, y_dev, exchange_ratio=float(ratio), random_seed=cfg.random_seed
        )
        print(
            f"  |A|=|B|={exchange.n_exchanged}  "
            f"base_train={len(exchange.base_train_idx)} "
            f"base_test={len(exchange.base_test_idx)} "
            f"meta_train={len(exchange.meta_train_idx)} "
            f"meta_test={len(exchange.meta_test_idx)}"
        )

        result = run_stacking(
            data.X_dev,
            y_dev,
            data.X_holdout,
            y_hold,
            exchange,
            base_model_name=cfg.base_model,
            meta_model_name=cfg.meta_model,
            oof_folds=cfg.oof_folds,
            random_seed=cfg.random_seed,
        )

        leak = validate_allocations(
            exchange,
            n_holdout=len(data.holdout),
            holdout_uids=data.holdout["sample_uid"].to_numpy(),
            development_uids=data.development["sample_uid"].to_numpy(),
            oof=result.oof,
        )
        for note in leak.notes:
            print(f"  [leakage] {note}")
        if not leak.ok:
            print("  LEAKAGE VIOLATIONS:")
            for v in leak.violations:
                print(f"    - {v}")
            leak.raise_if_failed()

        comps = report_compositions(
            exchange,
            dist_labels,
            cfg.distribution_key,
            cfg.target_similar_pct,
            cfg.target_dissimilar_pct,
        )
        for side, rep in comps.items():
            print(
                f"  {side} composition similar={rep.similar_pct:.1%} "
                f"dissimilar={rep.dissimilar_pct:.1%} "
                f"target_ok={rep.target_achieved}"
            )
            if rep.notes:
                print(f"    note: {rep.notes}")

        print(
            f"  base_test acc={result.base_test_metrics.accuracy:.4f}  "
            f"meta/stacking(dev) acc={result.meta_test_metrics.accuracy:.4f}  "
            f"holdout_stacking acc={result.holdout_stacking_metrics.accuracy:.4f}"
        )

        summary_rows.extend(
            metrics_rows(label, float(ratio), result, comps, leak)
        )
        cf = composition_to_frame(comps)
        cf.insert(0, "experiment", label)
        cf.insert(1, "exchange_ratio", float(ratio))
        composition_frames.append(cf)

        manifest = exchange.manifest(data.development)
        manifests[label] = manifest
        manifest.to_csv(out / f"manifest_{label}.csv", index=False)
        leakage_frames[label] = leakage_to_frame(leak)

        # Confusion matrix plots for holdout stacking
        plot_confusion(
            result.holdout_stacking_metrics.confusion,
            f"{label} holdout stacking",
            out / f"cm_holdout_stacking_{label}.png",
        )
        plot_confusion(
            result.base_test_metrics.confusion,
            f"{label} base test",
            out / f"cm_base_test_{label}.png",
        )

    summary = pd.DataFrame(summary_rows)
    compositions = pd.concat(composition_frames, ignore_index=True)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    compositions.to_csv(out / "compositions.csv", index=False)

    plot_comparison(summary, out / "holdout_stacking_comparison.png")
    xlsx = out / "cross_fold_stacking_results.xlsx"
    write_excel(
        xlsx,
        summary,
        compositions,
        manifests,
        leakage_frames,
        cfg.to_dict(),
    )

    # Pivot-friendly comparison table
    hold = summary[summary["metric_set"] == "holdout_stacking"][
        ["experiment", "exchange_ratio", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    hold.to_csv(out / "holdout_stacking_comparison.csv", index=False)
    print("=" * 72)
    print(f"Done. Results → {out}")
    print(hold.to_string(index=False))
    return out
