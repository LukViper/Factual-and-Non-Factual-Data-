#!/usr/bin/env bash
# Sequential Phase-1 runner (Default mode). Safe under low RAM / full swap.
set +e
cd "$(dirname "$0")"
PY="${PWD}/.venv/bin/python"
DATA="/home/luk_viper/ML-Project/training_data.csv"
LOG="logs/phase1_default.log"
mkdir -p logs outputs cache

FEATURES=(tfidf minilm bert roberta t5)
CLASSIFIERS=(
  "Logistic Regression"
  "Perceptron"
  "LDA"
  "Naive Bayes"
  "Decision Tree"
  "SVM"
  "KNN"
  "AdaBoost"
  "Gradient Boosting"
  "XGBoost"
  "LightGBM"
  "CatBoost"
  "Extra Trees"
  "Random Forest"
  "Neural Network (MLP)"
  "Voting"
  "Stacking"
)

echo "=== Phase1 start $(date) ===" | tee -a "$LOG"

for feat in "${FEATURES[@]}"; do
  echo "===== FEATURE ${feat} =====" | tee -a "$LOG"
  for clf in "${CLASSIFIERS[@]}"; do
    echo "[${feat}] START: ${clf}  $(date)" | tee -a "$LOG"
    "$PY" -u ML.py \
      --data "$DATA" \
      --sample 0 \
      --features "$feat" \
      --modes Default \
      --only "$clf" \
      --resume \
      --skip_cv \
      --n_iter 10 \
      >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[${feat}] FAILED: ${clf} (exit ${rc})  $(date)" | tee -a "$LOG"
    else
      echo "[${feat}] DONE: ${clf}  $(date)" | tee -a "$LOG"
    fi
    sync
    sleep 1
  done
done

echo "=== Building master misclassified workbook $(date) ===" | tee -a "$LOG"
"$PY" -u - <<'PY' | tee -a logs/phase1_default.log
from pathlib import Path
import pandas as pd
from ML import write_master_misclassified, OUT_DIR
frames, summaries = {}, []
for p in sorted(OUT_DIR.glob("always_misclassified_*.csv")):
    key = p.stem.replace("always_misclassified_", "")
    if key == "all_features":
        continue
    df = pd.read_csv(p)
    frames[key] = df
    test_n = None
    xlsx = OUT_DIR / f"results_{key}.xlsx"
    if xlsx.exists():
        try:
            info = pd.read_excel(xlsx, sheet_name="Dataset Info")
            d = dict(zip(info["Property"], info["Value"]))
            test_n = d.get("Test Size")
        except Exception:
            pass
    summaries.append({"Feature": key, "Always Misclassified Count": len(df), "Test Size": test_n})
write_master_misclassified(OUT_DIR / "always_misclassified_all_features.xlsx", frames, summaries)
print("Wrote", OUT_DIR / "always_misclassified_all_features.xlsx")
PY

echo "=== Phase1 done $(date) ===" | tee -a "$LOG"
