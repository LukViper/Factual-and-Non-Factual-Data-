#!/usr/bin/env bash
# Full real run: both balanced sizes × all embedders × 4 tuning modes.
# Resumable; writes results_full/ and FINAL_*.xlsx
set -euo pipefail
cd "$(dirname "$0")"
PY="${PWD}/.venv/bin/python"
DATA="${DATA:-/home/luk_viper/ML-Project/training_data.csv}"
MODES="${MODES:-Default,Grid Search,Random Search,Bayesian Optimization}"
FEATURES="${FEATURES:-tfidf,minilm,bert,roberta,t5}"
N_ITER="${N_ITER:-10}"
mkdir -p logs results_full cache_full

echo "=== FULL TUNING start $(date) ===" | tee -a logs/full_tuning.log

# --- 4k: full CV + all 4 modes ---
IFS=',' read -r -a FEATS <<< "$FEATURES"
for feat in "${FEATS[@]}"; do
  echo "===== 4k / $feat =====" | tee -a logs/full_tuning.log
  "$PY" -u ML.py --data "$DATA" --size 4k --features "$feat" \
    --modes "$MODES" --n_iter "$N_ITER" --resume \
    2>&1 | tee -a "logs/full_4k_${feat}.log"
done

# --- 67k: skip outer CV (memory), still all 4 modes ---
for feat in "${FEATS[@]}"; do
  echo "===== 67k / $feat =====" | tee -a logs/full_tuning.log
  "$PY" -u ML.py --data "$DATA" --size 67k --features "$feat" \
    --modes "$MODES" --n_iter "$N_ITER" --resume --skip_cv \
    2>&1 | tee -a "logs/full_67k_${feat}.log"
done

"$PY" -u ML.py --build-final
echo "=== FULL TUNING done $(date) ===" | tee -a logs/full_tuning.log
echo "Final: results_full/FINAL_all_results.xlsx"
echo "Final: results_full/FINAL_misclassified_ge5.xlsx"
