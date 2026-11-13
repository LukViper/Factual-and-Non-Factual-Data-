#!/usr/bin/env python3
"""Generate README_4k.md and README_67k.md from real experiment CSVs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RS = 42
EMB_ORDER = ["tfidf", "minilm", "bert", "roberta", "t5"]
MODE_ORDER = ["Default", "Grid Search", "Random Search", "Bayesian Optimization"]
CLF_ORDER = [
    "Logistic Regression", "Perceptron", "LDA", "Naive Bayes", "Decision Tree", "SVM", "KNN",
    "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Extra Trees",
    "Random Forest", "Neural Network (MLP)", "Voting", "Stacking",
]


def load_size(size: str) -> pd.DataFrame:
    frames = []
    for emb in EMB_ORDER:
        p = ROOT / "results_full" / size / emb / f"results_{emb}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["Balanced Size"] = size
            df["Embedder"] = emb
            rename = {
                "Training Accuracy *": "Training Accuracy",
                "Fit/Search Time (sec)": "Time (sec)",
                "ROC-AUC (Macro OvR)": "ROC-AUC",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            frames.append(df)

    for emb in EMB_ORDER:
        p = ROOT / "results_easy" / size / emb / "all_results.csv"
        if not p.exists():
            continue
        edf = pd.read_csv(p)
        edf["Balanced Size"] = size
        edf["Embedder"] = emb
        if frames:
            existing = pd.concat(frames, ignore_index=True)
            have = set(zip(existing["Embedder"], existing["Classifier Name"], existing["Tuning"]))
        else:
            have = set()
        keep = [r for _, r in edf.iterrows()
                if (emb, r["Classifier Name"], r["Tuning"]) not in have]
        if keep:
            frames.append(pd.DataFrame(keep))

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in [
        "Test Accuracy", "F1-Score (Macro)", "Precision (Macro)", "Recall (Macro)",
        "Training Accuracy", "Train-Test Gap", "Stratified K-Fold Mean", "ROC-AUC", "Time (sec)",
    ]:
        if c not in df.columns:
            df[c] = np.nan
    return df


def fmt(x, nd=4):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def md_table(df: pd.DataFrame, cols) -> str:
    if df is None or len(df) == 0:
        return "_No data yet._\n"
    use = [c for c in cols if c in df.columns]
    lines = [
        "| " + " | ".join(use) + " |",
        "| " + " | ".join(["---"] * len(use)) + " |",
    ]
    metric = {
        "Test Accuracy", "F1-Score (Macro)", "Precision (Macro)", "Recall (Macro)",
        "Training Accuracy", "Train-Test Gap", "Stratified K-Fold Mean", "ROC-AUC",
    }
    for _, r in df.iterrows():
        cells = []
        for c in use:
            v = r[c]
            if c in metric:
                cells.append(fmt(v))
            elif c == "Time (sec)":
                cells.append(fmt(v, 1) if not pd.isna(v) else "—")
            else:
                cells.append(str(v) if not pd.isna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def coverage(df: pd.DataFrame) -> str:
    lines = [
        "| Embedder | Classifiers | Tunings present | Rows | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for emb in EMB_ORDER:
        sub = df[df["Embedder"] == emb]
        if len(sub) == 0:
            lines.append(f"| `{emb}` | 0 | — | 0 | missing |")
            continue
        n_clf = sub["Classifier Name"].nunique()
        modes = [m for m in MODE_ORDER if m in set(sub["Tuning"])]
        rows = len(sub)
        expect = 17 * 4
        if rows >= expect and set(modes) == set(MODE_ORDER):
            status = "complete (4 tunings)"
        elif set(modes) == {"Default"} and n_clf >= 17:
            status = "Default only (Grid/Random/Bayesian pending)"
        else:
            status = f"partial ({rows}/{expect} expected)"
        lines.append(f"| `{emb}` | {n_clf} | {', '.join(modes)} | {rows} | {status} |")
    return "\n".join(lines) + "\n"


def best_overall(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Embedder", "Classifier Name", "Tuning", "Test Accuracy", "F1-Score (Macro)",
        "Precision (Macro)", "Recall (Macro)", "Overfitting Risk", "Fit Status",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.sort_values("Test Accuracy", ascending=False).head(15)[cols]


def best_per_embedder(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("Embedder")["Test Accuracy"].idxmax()
    cols = ["Embedder", "Classifier Name", "Tuning", "Test Accuracy", "F1-Score (Macro)", "Fit Status"]
    cols = [c for c in cols if c in df.columns]
    out = df.loc[idx, cols].copy()
    out["Embedder"] = pd.Categorical(out["Embedder"], EMB_ORDER)
    return out.sort_values("Embedder")


def pivot_test_acc(df: pd.DataFrame, emb: str) -> pd.DataFrame:
    sub = df[df["Embedder"] == emb].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    p = sub.pivot_table(index="Classifier Name", columns="Tuning", values="Test Accuracy", aggfunc="max")
    p = p.reindex(
        index=[c for c in CLF_ORDER if c in p.index],
        columns=[m for m in MODE_ORDER if m in p.columns],
    ).reset_index()
    return p


def ge5_summary(size: str) -> str:
    lines = [
        "| Embedder | Claims misclassified by ≥5 classifiers | Source |",
        "| --- | ---: | --- |",
    ]
    for emb in EMB_ORDER:
        path = None
        for cand in [
            ROOT / "results_full" / size / emb / f"misclassified_ge5_{emb}.csv",
            ROOT / "results_easy" / size / emb / "misclassified_ge5.csv",
        ]:
            if cand.exists() and len(pd.read_csv(cand)) > 0:
                path = cand
                break
        if path is None:
            am = ROOT / "results_easy" / size / emb / "always_misclassified.csv"
            if am.exists():
                n = len(pd.read_csv(am))
                lines.append(f"| `{emb}` | (always-misclassified: {n}; ge5 not yet) | `results_easy` |")
            else:
                lines.append(f"| `{emb}` | — | pending |")
            continue
        g = pd.read_csv(path)
        lines.append(f"| `{emb}` | {len(g)} | `{path.relative_to(ROOT)}` |")
    return "\n".join(lines) + "\n"


def write_readme(size: str, sample_desc: str, n_rows: str) -> Path:
    df = load_size(size)
    out = ROOT / f"README_{size}.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        f"# FEVER Results — Balanced **{size}**\n",
        f"_Auto-generated from real experiment CSVs on {now}. Values are measured, not synthetic._\n",
        "## Setup\n",
        "| Property | Value |",
        "| --- | --- |",
        f"| Balanced size | **{size}** — {sample_desc} |",
        f"| Rows used | {n_rows} |",
        "| Labels | `factual` (0) / `non-factual` (1) |",
        f"| Split | 80% train / 20% test, stratified, `random_state={RS}` |",
        "| Embedders | TF-IDF+SVD-500, MiniLM, BERT, RoBERTa, T5 |",
        "| Classifiers | 17 (LR, Perceptron, LDA, NB, DT, SVM, KNN, AdaBoost, GB, XGB, LGBM, CatBoost, ET, RF, MLP, Voting, Stacking) |",
        "| Tuning methods | Default, Grid Search, Random Search, Bayesian Optimization (Optuna TPE) |",
        "| Primary Excel | `results_full/FINAL_all_results.xlsx` |",
        "| ≥5 misclassified Excel | `results_full/FINAL_misclassified_ge5.xlsx` |",
        "",
        "## Coverage (what is finished)\n",
        coverage(df),
        "",
    ]

    if len(df) == 0:
        parts.append("_No result rows found yet for this size._\n")
        out.write_text("\n".join(parts))
        return out

    parts += [
        "## Best overall (top 15 by Test Accuracy)\n",
        md_table(best_overall(df), [
            "Embedder", "Classifier Name", "Tuning", "Test Accuracy", "F1-Score (Macro)",
            "Precision (Macro)", "Recall (Macro)", "Fit Status",
        ]),
        "",
        "## Best configuration per embedder\n",
        md_table(best_per_embedder(df), [
            "Embedder", "Classifier Name", "Tuning", "Test Accuracy", "F1-Score (Macro)", "Fit Status",
        ]),
        "",
        "## Best Test Accuracy by Embedder × Tuning\n",
    ]
    piv = df.pivot_table(index="Embedder", columns="Tuning", values="Test Accuracy", aggfunc="max")
    piv = piv.reindex(
        index=[e for e in EMB_ORDER if e in piv.index],
        columns=[m for m in MODE_ORDER if m in piv.columns],
    ).reset_index()
    parts += [md_table(piv, list(piv.columns)), ""]

    for emb in EMB_ORDER:
        sub = df[df["Embedder"] == emb]
        if len(sub) == 0:
            parts.append(f"## Embedder: `{emb}`\n\n_No results yet._\n")
            continue
        feat = emb
        if "Feature Set" in sub.columns and sub["Feature Set"].notna().any():
            feat = sub["Feature Set"].dropna().iloc[0]
        elif "Feature Name" in sub.columns and sub["Feature Name"].notna().any():
            feat = sub["Feature Name"].dropna().iloc[0]
        best = sub.loc[sub["Test Accuracy"].idxmax()]
        parts += [
            f"## Embedder: `{emb}`\n",
            f"Feature set: **{feat}**  \n",
            f'Result rows: **{len(sub)}** | Classifiers: **{sub["Classifier Name"].nunique()}** | '
            f'Tunings: **{", ".join(m for m in MODE_ORDER if m in set(sub["Tuning"]))}**\n',
            f'Best here: **{best["Classifier Name"]}** / **{best["Tuning"]}** — '
            f'Test Acc `{fmt(best["Test Accuracy"])}`, F1 `{fmt(best["F1-Score (Macro)"])}`\n',
            "### Test Accuracy by Classifier × Tuning\n",
            md_table(pivot_test_acc(df, emb), list(pivot_test_acc(df, emb).columns)),
            "### Full metrics (all classifier × tuning rows)\n",
        ]
        sub2 = sub.copy()
        sub2["_c"] = pd.Categorical(sub2["Classifier Name"], CLF_ORDER)
        sub2["_t"] = pd.Categorical(sub2["Tuning"], MODE_ORDER)
        sub2 = sub2.sort_values(["_c", "_t"])
        parts += [
            md_table(sub2, [
                "Classifier Name", "Tuning", "Test Accuracy", "F1-Score (Macro)",
                "Precision (Macro)", "Recall (Macro)", "Training Accuracy",
                "Train-Test Gap", "Stratified K-Fold Mean", "Overfitting Risk", "Fit Status",
            ]),
            "",
        ]

    parts += [
        "## Misclassified by ≥5 classifiers\n",
        "Claims on the held-out test set that were wrong for **at least 5** classifiers "
        "(best tuning per classifier). Sheets include **true_label vs predicted_label**.\n",
        ge5_summary(size),
        "",
        "## Artifact paths\n",
        "| Kind | Path |",
        "| --- | --- |",
        "| Combined results Excel | `results_full/FINAL_all_results.xlsx` |",
        "| ≥5 misclassified Excel | `results_full/FINAL_misclassified_ge5.xlsx` |",
        f"| Per-embedder (full tuning) | `results_full/{size}/<embedder>/results_<embedder>.xlsx` |",
        f"| Per-embedder (easy/Default) | `results_easy/{size}/<embedder>/results_{size}_<embedder>.xlsx` |",
        "| Master easy workbook | `results_easy/MASTER_all_sizes_embedders.xlsx` |",
        "",
        "## How to reproduce\n",
        "```bash",
        "source .venv/bin/activate",
        f"python ML.py --size {size} --features tfidf,minilm,bert,roberta,t5 \\",
        '  --modes "Default,Grid Search,Random Search,Bayesian Optimization" --resume'
        + (" --skip_cv" if size == "67k" else ""),
        "python ML.py --build-final",
        "python generate_size_readmes.py",
        "```\n",
    ]
    out.write_text("\n".join(parts))
    print(f"Wrote {out} ({out.stat().st_size} bytes), data rows={len(df)}")
    return out


if __name__ == "__main__":
    write_readme("4k", "2,000 factual + 2,000 non-factual", "~4,000")
    write_readme("67k", "full minority-class balance (~33,567 + 33,567)", "~67,134")
