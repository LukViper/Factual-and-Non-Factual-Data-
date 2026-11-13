#!/usr/bin/env python3
"""
Cross-Fold Data-Exchange Stacking  (faculty demo — one file)
============================================================

What this program does (explain in viva / report):

  1. Load BALANCED FEVER claims (equal factual / non-factual).
  2. Split: 80% DEVELOPMENT  +  20% HOLD OUT (never used in training).
  3. Split development into 5 equal stratified folds: F1 F2 F3 F4 F5.
  4. Assign Base Model and Meta Model roles (see diagram below).
  5. Optionally EXCHANGE a few samples between F1 and F5 (E0–E3).
  6. Train Base → get probability scores (out-of-fold where needed).
  7. Train Meta on those scores → final factual / non-factual prediction.
  8. Evaluate on the untouched 20% holdout; save Excel.

Run examples:
  python cross_fold_easy.py --size 4k  --embedder tfidf
  python cross_fold_easy.py --size 4k  --embedder minilm --ratio 0.25
  python cross_fold_easy.py --size 67k --embedder bert --all-ratios

Visual overview (data roles AFTER exchange of subsets A⊂F1 and B⊂F5)
--------------------------------------------------------------------

  DEVELOPMENT (80%)                         HOLD OUT (20%)
  ┌─────────────────────────────────────┐   ┌──────────────┐
  │  F1\\A │ A │ F2 │ F3 │ F4 │ F5\\B │ B │   │  NEVER TRAIN │
  └─────────────────────────────────────┘   └──────────────┘

  Base Model:
    TRAIN  = (F1 without A) + F2 + F3 + F4 + B
    TEST   = A + (F5 without B)

  Meta Model:
    TRAIN  = A + F2 + F3 + F4 + (F5 without B)
    TEST   = (F1 without A) + B

  Stacking:
    Meta features = Base Model probability scores
    (out-of-fold if that sample was in Base TRAIN — no leakage)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_CSV = "/home/luk_viper/ML-Project/training_data.csv"
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_cross_fold" / "faculty_demo"

SIZE_TO_SAMPLE = {
    "4k": 4000,  # 2000 + 2000
    "67k": 0,  # full balanced (~67k)
}

# Experiments faculty asked for
ALL_RATIOS = [0.0, 0.10, 0.25, 0.40]
EXP_NAME = {0.0: "E0", 0.10: "E1", 0.25: "E2", 0.40: "E3"}


# ---------------------------------------------------------------------------
# STEP 1 — load balanced FEVER claims
# ---------------------------------------------------------------------------
def load_balanced(sample_total: int) -> pd.DataFrame:
    """Equal number of factual and non-factual claims."""
    df = pd.read_csv(DATA_CSV, usecols=["id", "claim", "verifiable"])
    df = df.drop_duplicates(subset="claim").dropna(subset=["claim"])
    df["claim"] = df["claim"].astype(str).str.strip()
    df = df[df["claim"] != ""].copy()

    per_class = int(df["verifiable"].value_counts().min())
    if sample_total > 0:
        per_class = min(per_class, sample_total // 2)

    bal = (
        df.groupby("verifiable", group_keys=False)
        .sample(n=per_class, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    bal["label"] = (bal["verifiable"] == "non-factual").astype(int)
    print(
        f"[1] Balanced data: {len(bal)} rows "
        f"(factual={(bal.label == 0).sum()}, non-factual={(bal.label == 1).sum()})"
    )
    return bal


# ---------------------------------------------------------------------------
# STEP 2 — features (reuse project pipeline: TF-IDF / MiniLM / BERT / …)
# ---------------------------------------------------------------------------
def build_features(embedder: str, sample_total: int):
    """
    80/20 split + features.
    Holdout is never used to fit TF-IDF/SVD or the scaler.
    """
    from cross_fold_stacking.data import prepare_dataset

    data = prepare_dataset(
        data_csv=DATA_CSV,
        sample_total=sample_total,
        holdout_fraction=0.20,
        random_seed=RANDOM_STATE,
        embedder=embedder,
        svd_dim=500,
    )
    print(
        f"[2] Development={len(data.development)}  Holdout={len(data.holdout)}  "
        f"Features={data.feature_name} {data.X_dev.shape}"
    )
    return data


# ---------------------------------------------------------------------------
# STEP 3 — five equal folds on development
# ---------------------------------------------------------------------------
def make_five_folds(y: np.ndarray) -> np.ndarray:
    """
    fold_id[i] ∈ {1,2,3,4,5}
    Each fold ≈ 20% of development (stratified).
    """
    fold_id = np.zeros(len(y), dtype=np.int8)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        fold_id[te] = k
    for k in range(1, 6):
        print(f"     F{k}: {(fold_id == k).sum()} samples")
    print("[3] Five stratified folds ready")
    return fold_id


# ---------------------------------------------------------------------------
# STEP 4 + 5 — Base / Meta roles + optional F1↔F5 sample exchange
# ---------------------------------------------------------------------------
def allocate_with_exchange(fold_id: np.ndarray, y: np.ndarray, ratio: float):
    """
    Initial roles (before exchange):

      Fold | Base        | Meta
      F1   | TRAIN       | TEST
      F2   | TRAIN       | TRAIN   ← shared
      F3   | TRAIN       | TRAIN   ← shared
      F4   | TRAIN       | TRAIN   ← shared
      F5   | TEST        | TRAIN

    Exchange: take subset A from F1 and subset B from F5 (|A|=|B|).
    Swap their Base/Meta roles (sample-level, not whole folds).
    """
    from cross_fold_stacking.folds import FoldAllocation
    from cross_fold_stacking.exchange import apply_cross_fold_exchange

    fold_indices = {f: np.where(fold_id == f)[0] for f in range(1, 6)}
    folds = FoldAllocation(fold_id=fold_id, fold_indices=fold_indices)
    ex = apply_cross_fold_exchange(folds, y, exchange_ratio=ratio, random_seed=RANDOM_STATE)

    print(
        f"[4-5] Exchange {ratio:.0%} → |A|=|B|={ex.n_exchanged}\n"
        f"      Base  train={len(ex.base_train_idx)}  test={len(ex.base_test_idx)}\n"
        f"      Meta  train={len(ex.meta_train_idx)}  test={len(ex.meta_test_idx)}"
    )
    return ex


# ---------------------------------------------------------------------------
# STEP 6 — Base model + out-of-fold scores (no leakage into Meta)
# ---------------------------------------------------------------------------
def base_scores_oof(X, y, exchange):
    """
    Base = Logistic Regression → probability of non-factual.

    Why OOF?
      F2,F3,F4 are training for BOTH Base and Meta.
      If Meta saw Base's *in-sample* scores on those rows → leakage.
      So for any row that Base trained on, we predict with a clone
      that did NOT see that row (inner 5-fold CV on Base train).
    """
    from cross_fold_stacking.oof import generate_base_oof_scores
    from cross_fold_stacking.models import make_base_model

    base = make_base_model("logistic", RANDOM_STATE)
    oof = generate_base_oof_scores(
        X, y, exchange, base, oof_folds=5, random_seed=RANDOM_STATE
    )
    print("[6] Base model trained; OOF probability scores ready for Meta")
    return oof


# ---------------------------------------------------------------------------
# STEP 7 — Meta model (stacking) + STEP 8 — evaluate
# ---------------------------------------------------------------------------
def metrics(name: str, y_true, y_pred, scores) -> dict:
    auc = None
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "set": name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def run_one_ratio(data, fold_id, ratio: float) -> tuple[dict, pd.DataFrame]:
    """Full Base → Meta stacking for one exchange ratio."""
    y = data.development["label"].to_numpy()
    X = data.X_dev
    y_h = data.holdout["label"].to_numpy()
    X_h = data.X_holdout

    print("-" * 60)
    print(f"Experiment {EXP_NAME.get(ratio, '?')}  exchange={ratio:.0%}")
    exchange = allocate_with_exchange(fold_id, y, ratio)
    oof = base_scores_oof(X, y, exchange)

    # --- Base accuracy on its own test split ---
    from cross_fold_stacking.models import predict_scores

    bt = exchange.base_test_idx
    y_bt, s_bt = predict_scores(oof.base_fitted, X[bt])
    base_m = metrics("base_test", y[bt], y_bt, s_bt)

    # --- Meta trains on Base OOF scores ---
    mt, mte = exchange.meta_train_idx, exchange.meta_test_idx
    meta = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    meta.fit(oof.scores[mt].reshape(-1, 1), y[mt])
    y_mt, s_mt = predict_scores(meta, oof.scores[mte].reshape(-1, 1))
    meta_m = metrics("meta_test_stacking", y[mte], y_mt, s_mt)
    print("[7] Meta model (stacking) trained on Base OOF scores")

    # --- Untouched holdout ---
    y_hb, s_hb = predict_scores(oof.base_fitted, X_h)
    hold_base = metrics("holdout_base", y_h, y_hb, s_hb)
    y_hs, s_hs = predict_scores(meta, s_hb.reshape(-1, 1))
    hold_stack = metrics("holdout_stacking", y_h, y_hs, s_hs)
    print(
        f"[8] Holdout stacking  acc={hold_stack['accuracy']:.4f}  "
        f"F1={hold_stack['f1']:.4f}  AUC={hold_stack['roc_auc']:.4f}"
    )

    rows = []
    for m in (base_m, meta_m, hold_base, hold_stack):
        rows.append(
            {
                "experiment": EXP_NAME.get(ratio, f"r{ratio}"),
                "exchange_ratio": ratio,
                "n_exchanged_each": exchange.n_exchanged,
                "base_train": len(exchange.base_train_idx),
                "base_test": len(exchange.base_test_idx),
                "meta_train": len(exchange.meta_train_idx),
                "meta_test": len(exchange.meta_test_idx),
                **m,
            }
        )

    # Sample allocation table (for faculty: who saw what)
    manifest = exchange.manifest(data.development)[
        [
            "sample_uid",
            "id",
            "claim",
            "label",
            "original_fold",
            "exchanged",
            "base_role",
            "meta_role",
            "cross_model_train_exposure",
        ]
    ]
    return rows, manifest


# ---------------------------------------------------------------------------
# Excel + ASCII diagram for the report
# ---------------------------------------------------------------------------
ROLE_DIAGRAM = """
INITIAL ALLOCATION (before exchange)
====================================
         Base Model          Meta Model
  F1     TRAIN               TEST
  F2     TRAIN               TRAIN     } shared training
  F3     TRAIN               TRAIN     } (needs OOF scores)
  F4     TRAIN               TRAIN     }
  F5     TEST                TRAIN

AFTER SAMPLE EXCHANGE (A from F1  ↔  B from F5)
===============================================
  Data                Base              Meta
  F1 without A        TRAIN             TEST
  A (from F1)         TEST              TRAIN   ← swapped
  F2                  TRAIN             TRAIN
  F3                  TRAIN             TRAIN
  F4                  TRAIN             TRAIN
  F5 without B        TEST              TRAIN
  B (from F5)         TRAIN             TEST    ← swapped

STACKING FLOW
=============
  Development claims
        │
        ▼
  ┌─────────────┐     OOF probability      ┌─────────────┐
  │  BASE MODEL │ ───────────────────────► │  META MODEL │ ──► final label
  │  (LogReg)   │     scores (1 number)    │  (LogReg)   │
  └─────────────┘                          └─────────────┘
        │                                         │
        └──────────── holdout features ───────────┘
                      (Base never saw holdout;
                       Meta uses Base scores only)
"""


def save_excel(summary: pd.DataFrame, manifests: dict, out: Path, embedder: str, size: str):
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"faculty_demo_{size}_{embedder}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        hold = summary[summary["set"] == "holdout_stacking"]
        hold.to_excel(xl, sheet_name="Holdout_Stacking", index=False)
        pd.DataFrame({"diagram": ROLE_DIAGRAM.splitlines()}).to_excel(
            xl, sheet_name="Diagram", index=False
        )
        for name, mf in manifests.items():
            # keep Excel light: first 500 rows of manifest
            mf.head(500).to_excel(xl, sheet_name=f"Manifest_{name}"[:31], index=False)
    print(f"\nSaved Excel → {path}")
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Faculty demo: cross-fold exchange stacking")
    p.add_argument("--size", choices=["4k", "67k"], default="4k")
    p.add_argument(
        "--embedder",
        choices=["tfidf", "minilm", "bert", "roberta", "t5"],
        default="tfidf",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Single exchange ratio (0, 0.1, 0.25, 0.4). Default with --all-ratios.",
    )
    p.add_argument(
        "--all-ratios",
        action="store_true",
        help="Run E0-E3 (0/10/25/40 percent exchange)",
    )
    args = p.parse_args()

    print(ROLE_DIAGRAM)
    sample_total = SIZE_TO_SAMPLE[args.size]
    data = build_features(args.embedder, sample_total)
    y = data.development["label"].to_numpy()
    fold_id = make_five_folds(y)

    if args.all_ratios or args.ratio is None:
        ratios = ALL_RATIOS
    else:
        ratios = [args.ratio]

    all_rows = []
    manifests = {}
    for r in ratios:
        rows, mf = run_one_ratio(data, fold_id, float(r))
        all_rows.extend(rows)
        manifests[EXP_NAME.get(r, f"r{r}")] = mf

    summary = pd.DataFrame(all_rows)
    path = save_excel(summary, manifests, OUT_DIR, args.embedder, args.size)

    hold = summary[summary["set"] == "holdout_stacking"][
        ["experiment", "exchange_ratio", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    print("\nHoldout stacking comparison:")
    print(hold.to_string(index=False))
    print(f"\nDone. Open: {path}")


if __name__ == "__main__":
    main()
