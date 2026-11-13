#!/usr/bin/env python3
"""
Merge real per-size / per-embedder results into final Excel workbooks.

Writes:
  results_full/FINAL_all_results.xlsx
  results_full/FINAL_misclassified_ge5.xlsx
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results_full"


def _collect_results(out_root: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(out_root.glob("*/*/results_*.csv")):
        size = csv.parent.parent.name
        emb = csv.parent.name
        df = pd.read_csv(csv)
        if "Balanced Size" not in df.columns:
            df.insert(0, "Balanced Size", size)
        if "Embedder" not in df.columns:
            df.insert(1, "Embedder", emb)
        rows.append(df)
    if not rows:
        # also accept results_easy layout
        easy = ROOT / "results_easy"
        for csv in sorted(easy.glob("*/*/all_results.csv")):
            size = csv.parent.parent.name
            emb = csv.parent.name
            df = pd.read_csv(csv)
            if "Balanced Size" not in df.columns:
                df.insert(0, "Balanced Size", size)
            if "Embedder" not in df.columns:
                df.insert(1, "Embedder", emb)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _collect_ge5(out_root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for csv in sorted(out_root.glob("*/*/misclassified_ge5_*.csv")):
        size = csv.parent.parent.name
        emb = csv.parent.name
        df = pd.read_csv(csv)
        if len(df) == 0:
            continue
        df.insert(0, "Balanced Size", size)
        df.insert(1, "Embedder", emb)
        frames[f"{size}_{emb}"] = df
    return frames


def build_all(out_root: Path | None = None, min_wrong: int = 5) -> dict:
    out_root = Path(out_root or DEFAULT_OUT)
    out_root.mkdir(parents=True, exist_ok=True)

    all_df = _collect_results(out_root)
    ge5_frames = _collect_ge5(out_root)

    results_path = out_root / "FINAL_all_results.xlsx"
    ge5_path = out_root / "FINAL_misclassified_ge5.xlsx"

    summary = []
    if len(all_df):
        for (size, emb), g in all_df.groupby(["Balanced Size", "Embedder"], sort=True):
            summary.append({
                "Balanced Size": size,
                "Embedder": emb,
                "Result Rows": len(g),
                "Classifiers": g["Classifier Name"].nunique() if "Classifier Name" in g else None,
                "Tunings": ", ".join(sorted(g["Tuning"].astype(str).unique())) if "Tuning" in g else None,
                "Best Test Accuracy": g["Test Accuracy"].max() if "Test Accuracy" in g else None,
                "Best F1 Macro": g["F1-Score (Macro)"].max() if "F1-Score (Macro)" in g else None,
            })

    with pd.ExcelWriter(results_path, engine="openpyxl") as xw:
        pd.DataFrame(summary).to_excel(xw, sheet_name="Summary", index=False)
        if len(all_df):
            all_df.to_excel(xw, sheet_name="All Results", index=False)
            # best config per size x embedder x classifier
            tmp = all_df.copy()
            if "Overfitting Risk" in tmp.columns:
                tmp["_high"] = (tmp["Overfitting Risk"] == "HIGH").astype(int)
            else:
                tmp["_high"] = 0
            sort_cols = ["Balanced Size", "Embedder", "Classifier Name", "_high"]
            asc = [True, True, True, True]
            if "Stratified K-Fold Mean" in tmp.columns:
                sort_cols.append("Stratified K-Fold Mean"); asc.append(False)
            if "Test Accuracy" in tmp.columns:
                sort_cols.append("Test Accuracy"); asc.append(False)
            best = (
                tmp.sort_values(sort_cols, ascending=asc)
                .groupby(["Balanced Size", "Embedder", "Classifier Name"], sort=False)
                .head(1)
                .drop(columns="_high")
                .reset_index(drop=True)
            )
            best.to_excel(xw, sheet_name="Best per Clf", index=False)
            try:
                piv = all_df.pivot_table(
                    index=["Balanced Size", "Embedder", "Classifier Name"],
                    columns="Tuning",
                    values="Test Accuracy",
                    aggfunc="max",
                ).reset_index()
                piv.to_excel(xw, sheet_name="TestAcc by Tuning", index=False)
            except Exception:
                pass
            for (size, emb), g in all_df.groupby(["Balanced Size", "Embedder"], sort=True):
                g.to_excel(xw, sheet_name=f"{size}_{emb}"[:31], index=False)

    ge5_summary = []
    with pd.ExcelWriter(ge5_path, engine="openpyxl") as xw:
        if ge5_frames:
            for key, df in ge5_frames.items():
                if len(df) == 0:
                    continue
                ge5_summary.append({
                    "Key": key,
                    "Balanced Size": df["Balanced Size"].iloc[0],
                    "Embedder": df["Embedder"].iloc[0],
                    f"N Misclassified by >={min_wrong}": len(df),
                })
            if not ge5_summary:
                pd.DataFrame({
                    "note": [f"ge5 CSVs exist but none have rows (need >= {min_wrong} classifiers wrong)."]
                }).to_excel(xw, sheet_name="Summary", index=False)
            else:
                pd.DataFrame(ge5_summary).to_excel(xw, sheet_name="Summary", index=False)
                combined = pd.concat(
                    [df for df in ge5_frames.values() if len(df)], ignore_index=True
                )
                lean_cols = [c for c in [
                    "Balanced Size", "Embedder", "id", "claim",
                    "true_label", "predicted_label",
                    "n_classifiers_wrong", "n_classifiers_total",
                    "classifiers_that_misclassified",
                ] if c in combined.columns]
                combined[lean_cols].to_excel(xw, sheet_name="True vs Predicted", index=False)
                for key, df in ge5_frames.items():
                    if len(df):
                        df.to_excel(xw, sheet_name=key[:31], index=False)
        else:
            pd.DataFrame({
                "note": [
                    f"No non-empty misclassified_ge5_*.csv under {out_root}. "
                    "Run full tuning so >=5 classifiers have predictions."
                ]
            }).to_excel(xw, sheet_name="Summary", index=False)

    print(f"Wrote {results_path}")
    print(f"Wrote {ge5_path}")
    return {"results": str(results_path), "misclassified_ge5": str(ge5_path)}


if __name__ == "__main__":
    build_all()
