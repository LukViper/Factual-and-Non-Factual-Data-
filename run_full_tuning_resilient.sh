#!/usr/bin/env bash
# One classifier × one feature at a time. Survives single-job failures / OOMs.
set +e
cd "$(dirname "$0")"
PY="${PWD}/.venv/bin/python"
DATA="${DATA:-/home/luk_viper/ML-Project/training_data.csv}"
MODES="${MODES:-Default,Grid Search,Random Search,Bayesian Optimization}"
N_ITER="${N_ITER:-10}"
LOG="logs/full_tuning_resilient.log"
mkdir -p logs results_full cache_full

FEATURES=(tfidf minilm bert roberta t5)
SIZES=(4k 67k)
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

echo "=== RESILIENT FULL TUNING start $(date) ===" | tee -a "$LOG"

for size in "${SIZES[@]}"; do
  SKIP=""
  if [[ "$size" == "67k" ]]; then
    SKIP="--skip_cv"
  fi
  for feat in "${FEATURES[@]}"; do
    for clf in "${CLASSIFIERS[@]}"; do
      echo "[$size/$feat] START: $clf  $(date)" | tee -a "$LOG"
      "$PY" -u ML.py --data "$DATA" --size "$size" --features "$feat" \
        --modes "$MODES" --only "$clf" --n_iter "$N_ITER" --resume $SKIP \
        >> "logs/full_${size}_${feat}.log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "[$size/$feat] FAILED: $clf (exit $rc)  $(date)" | tee -a "$LOG"
      else
        echo "[$size/$feat] DONE: $clf  $(date)" | tee -a "$LOG"
      fi
      sync
      sleep 1
    done
  done
done

"$PY" -u ML.py --build-final >> "$LOG" 2>&1
echo "=== RESILIENT FULL TUNING done $(date) ===" | tee -a "$LOG"
echo "Final: results_full/FINAL_all_results.xlsx"
echo "Final: results_full/FINAL_misclassified_ge5.xlsx"
