#!/usr/bin/env bash
# Run cross-fold exchange stacking for all sizes × embedders.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# HF downloads often blocked here; DistilBERT local fallbacks are used for bert/roberta/t5.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

SIZES=(4k 67k)
EMBEDDERS=(tfidf minilm bert roberta t5)
LOG_DIR=results_cross_fold/logs
mkdir -p "$LOG_DIR"

for size in "${SIZES[@]}"; do
  for emb in "${EMBEDDERS[@]}"; do
    name="cross_fold_stacking_${size}_${emb}"
    out="results_cross_fold/${name}"
    if [[ -f "${out}/cross_fold_stacking_results.xlsx" && "${FORCE:-0}" != "1" ]]; then
      echo "[skip] ${name} (xlsx exists; FORCE=1 to redo)"
      continue
    fi
    echo "================================================================"
    echo "[run] ${name}  $(date -Is)"
    echo "================================================================"
    python -m cross_fold_stacking \
      --size "$size" \
      --embedder "$emb" \
      --name "$name" \
      --output-dir results_cross_fold \
      2>&1 | tee "${LOG_DIR}/${name}.log"
  done
done

python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("results_cross_fold")
runs = sorted(root.glob("cross_fold_stacking_*_*"))
summaries, comps, holdouts = [], [], []
master = root / "MASTER_cross_fold_stacking.xlsx"

with pd.ExcelWriter(master, engine="openpyxl") as xl:
    for d in runs:
        xlsx = d / "cross_fold_stacking_results.xlsx"
        if not xlsx.exists():
            continue
        key = d.name.replace("cross_fold_stacking_", "")
        summary = pd.read_excel(xlsx, sheet_name="Summary")
        summary.insert(0, "run", key)
        # parse size/embedder
        parts = key.split("_", 1)
        size = parts[0] if parts else ""
        emb = parts[1] if len(parts) > 1 else ""
        summary.insert(1, "size", size)
        summary.insert(2, "embedder", emb)
        summaries.append(summary)
        summary.to_excel(xl, sheet_name=f"S_{key}"[:31], index=False)
        try:
            comp = pd.read_excel(xlsx, sheet_name="Compositions")
            comp.insert(0, "run", key)
            comps.append(comp)
        except Exception:
            pass
        hold = summary[summary["metric_set"] == "holdout_stacking"][
            ["run", "size", "embedder", "experiment", "exchange_ratio",
             "accuracy", "precision", "recall", "f1", "roc_auc"]
        ]
        holdouts.append(hold)

    if not summaries:
        raise SystemExit("No result workbooks found")
    all_sum = pd.concat(summaries, ignore_index=True)
    all_sum.to_excel(xl, sheet_name="Summary_All", index=False)
    if comps:
        pd.concat(comps, ignore_index=True).to_excel(xl, sheet_name="Compositions_All", index=False)
    h_all = pd.concat(holdouts, ignore_index=True)
    h_all.to_excel(xl, sheet_name="Holdout_Stacking_All", index=False)
    piv = h_all.pivot_table(
        index=["size", "experiment"], columns="embedder", values="accuracy", aggfunc="first"
    )
    piv.to_excel(xl, sheet_name="Acc_Size_Exp_x_Emb")
    best = (
        h_all.sort_values("accuracy", ascending=False)
        .groupby(["size", "embedder"], as_index=False)
        .first()[["size", "embedder", "experiment", "exchange_ratio", "accuracy", "f1", "roc_auc"]]
    )
    best.to_excel(xl, sheet_name="BestHoldout_by_SizeEmb", index=False)
    print("Wrote", master.resolve())
    print(h_all.to_string(index=False))
PY

echo "ALL DONE $(date -Is)"
