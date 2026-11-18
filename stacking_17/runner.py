"""Run normal + cross-fold 17-base stacking; write Excel workbooks."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cross_fold_stacking.data import prepare_dataset

from .cross_fold_stack import run_cross_fold_stacking
from .normal_stack import run_normal_stacking

OUT_ROOT = Path(__file__).resolve().parent.parent / "results_stacking_17"


def _write_run_excel(
    path: Path,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    config: pd.DataFrame,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        ranking.to_excel(xl, sheet_name="BaseRanking", index=False)
        config.to_excel(xl, sheet_name="Config", index=False)
        # convenient pivots
        if "metric_set" in summary.columns:
            hold = summary[
                summary["metric_set"].isin(["holdout", "holdout_stacking"])
            ].copy()
            if not hold.empty:
                hold.to_excel(xl, sheet_name="Holdout", index=False)
        if extra:
            pd.DataFrame([extra]).to_excel(xl, sheet_name="RunInfo", index=False)


def run_one(
    size: str,
    embedder: str,
    *,
    top_k_meta: int = 3,
    exchange_ratios: list[float] | None = None,
    data_csv: str = "/home/luk_viper/ML-Project/training_data.csv",
    random_state: int = 42,
) -> Path:
    sample_total = 4000 if size == "4k" else 0
    out_dir = OUT_ROOT / f"{size}_{embedder}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"STACKING-17  size={size}  embedder={embedder}")
    print("=" * 72)

    data = prepare_dataset(
        data_csv=data_csv,
        sample_total=sample_total,
        holdout_fraction=0.20,
        random_seed=random_state,
        embedder=embedder,
        svd_dim=500,
    )
    y_dev = data.development["label"].to_numpy()
    y_hold = data.holdout["label"].to_numpy()
    info = {
        "size": size,
        "embedder": embedder,
        "feature_name": data.feature_name,
        "n_dev": len(data.development),
        "n_holdout": len(data.holdout),
        "X_dev_shape": str(data.X_dev.shape),
        "top_k_meta": top_k_meta,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    # --- Normal stacking ---
    n_sum, n_rank, n_cfg = run_normal_stacking(
        data.X_dev,
        y_dev,
        data.X_holdout,
        y_hold,
        size_label=size,
        embedder=embedder,
        feature_name=data.feature_name,
        random_state=random_state,
        top_k_meta=top_k_meta,
    )
    _write_run_excel(
        out_dir / "normal_stacking.xlsx", n_sum, n_rank, n_cfg, info
    )

    # --- Cross-fold stacking ---
    c_sum, c_rank, c_cfg = run_cross_fold_stacking(
        data.X_dev,
        y_dev,
        data.X_holdout,
        y_hold,
        size_label=size,
        embedder=embedder,
        feature_name=data.feature_name,
        exchange_ratios=exchange_ratios or [0.0, 0.10, 0.25, 0.40],
        random_state=random_state,
        top_k_meta=top_k_meta,
    )
    _write_run_excel(
        out_dir / "cross_fold_stacking.xlsx", c_sum, c_rank, c_cfg, info
    )

    # Combined workbook for this size×embedder
    combined = out_dir / f"combined_{size}_{embedder}.xlsx"
    with pd.ExcelWriter(combined, engine="openpyxl") as xl:
        n_sum.to_excel(xl, sheet_name="Normal_Summary", index=False)
        c_sum.to_excel(xl, sheet_name="CrossFold_Summary", index=False)
        n_rank.to_excel(xl, sheet_name="Normal_Ranking", index=False)
        c_rank.to_excel(xl, sheet_name="CrossFold_Ranking", index=False)
        n_hold = n_sum[n_sum["metric_set"] == "holdout"]
        c_hold = c_sum[c_sum["metric_set"] == "holdout_stacking"]
        n_hold.to_excel(xl, sheet_name="Normal_Holdout", index=False)
        c_hold.to_excel(xl, sheet_name="CrossFold_Holdout", index=False)
        # best meta comparison
        cmp_rows = []
        if not n_hold.empty:
            best_n = n_hold.sort_values("accuracy", ascending=False).iloc[0]
            cmp_rows.append(
                {
                    "method": "normal_stacking",
                    "meta_model": best_n["meta_model"],
                    "accuracy": best_n["accuracy"],
                    "f1": best_n["f1"],
                    "roc_auc": best_n["roc_auc"],
                    "experiment": "—",
                    "exchange_ratio": None,
                }
            )
        if not c_hold.empty:
            best_c = c_hold.sort_values("accuracy", ascending=False).iloc[0]
            cmp_rows.append(
                {
                    "method": "cross_fold_stacking",
                    "meta_model": best_c["meta_model"],
                    "accuracy": best_c["accuracy"],
                    "f1": best_c["f1"],
                    "roc_auc": best_c["roc_auc"],
                    "experiment": best_c["experiment"],
                    "exchange_ratio": best_c["exchange_ratio"],
                }
            )
        pd.DataFrame(cmp_rows).to_excel(xl, sheet_name="BestHoldout_Compare", index=False)
        pd.DataFrame([info]).to_excel(xl, sheet_name="RunInfo", index=False)

    print(f"Wrote {combined}")
    return out_dir


def build_master() -> Path:
    """Merge all combined workbooks into one master Excel."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    normals, crosses, ranks, compares = [], [], [], []
    for d in sorted(OUT_ROOT.glob("*_*")):
        if not d.is_dir():
            continue
        n = d / "normal_stacking.xlsx"
        c = d / "cross_fold_stacking.xlsx"
        if n.exists():
            normals.append(pd.read_excel(n, sheet_name="Summary"))
            ranks.append(pd.read_excel(n, sheet_name="BaseRanking"))
        if c.exists():
            crosses.append(pd.read_excel(c, sheet_name="Summary"))
            ranks.append(pd.read_excel(c, sheet_name="BaseRanking"))
        comb = list(d.glob("combined_*.xlsx"))
        if comb:
            try:
                cmp = pd.read_excel(comb[0], sheet_name="BestHoldout_Compare")
                parts = d.name.split("_", 1)
                cmp.insert(0, "size", parts[0] if parts else "")
                cmp.insert(1, "embedder", parts[1] if len(parts) > 1 else "")
                compares.append(cmp)
            except Exception:
                pass

    master = OUT_ROOT / "MASTER_stacking_17.xlsx"
    with pd.ExcelWriter(master, engine="openpyxl") as xl:
        if normals:
            ns = pd.concat(normals, ignore_index=True)
            ns.to_excel(xl, sheet_name="Normal_All", index=False)
            ns[ns["metric_set"] == "holdout"].to_excel(
                xl, sheet_name="Normal_Holdout_All", index=False
            )
        if crosses:
            cs = pd.concat(crosses, ignore_index=True)
            cs.to_excel(xl, sheet_name="CrossFold_All", index=False)
            cs[cs["metric_set"] == "holdout_stacking"].to_excel(
                xl, sheet_name="CrossFold_Holdout_All", index=False
            )
        if ranks:
            pd.concat(ranks, ignore_index=True).to_excel(
                xl, sheet_name="Rankings_All", index=False
            )
        if compares:
            cmp = pd.concat(compares, ignore_index=True)
            # attach size/embedder from folder if missing
            cmp.to_excel(xl, sheet_name="Best_Compare_All", index=False)
        # note
        pd.DataFrame(
            [
                {
                    "bases_4k": "15 individual classifiers from the 17-suite "
                    "(Voting & nested Stacking excluded as bases)",
                    "bases_67k": "Same suite minus KNN & MLP (too slow on ~50k); "
                    "Gradient Boosting uses HistGradientBoosting",
                    "meta": "Top-K validation-ranked classifiers from the same pool",
                    "methods": "normal_stacking + cross_fold_stacking",
                    "split": "Same 80% development / 20% holdout for both methods",
                }
            ]
        ).to_excel(xl, sheet_name="DesignNote", index=False)
    print(f"MASTER → {master}")
    return master
