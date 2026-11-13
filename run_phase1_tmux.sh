#!/usr/bin/env bash
# Durable Phase-1 runner: one classifier at a time inside tmux.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PWD}/.venv/bin/python"
DATA="/home/luk_viper/ML-Project/training_data.csv"
FEATURES_CSV="${FEATURES:-tfidf,minilm,bert,roberta,t5}"
MODES="${MODES:-Default}"
SESSION="${SESSION:-fever_ml}"

mkdir -p logs outputs cache

CLASSIFIERS=(
  "Logistic Regression"
  "Perceptron"
  "LDA"
  "KNN"
  "Decision Tree"
  "Extra Trees"
  "Random Forest"
  "XGBoost"
  "CatBoost"
  "AdaBoost"
  "Gradient Boosting"
  "LightGBM"
  "Voting"
  "Stacking"
  "SVM"
  "Naive Bayes"
  "Neural Network (MLP)"
)

IFS=',' read -r -a FEATURE_ARR <<< "$FEATURES_CSV"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 0
fi

SCRIPT="logs/_phase1_commands.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set +e"
  echo "cd $(pwd)"
  echo "PY=\"$PY\""
  echo "echo \"=== Phase1 start \$(date) ===\" | tee -a logs/phase1_default.log"
  for feat in "${FEATURE_ARR[@]}"; do
    echo "echo \"===== FEATURE ${feat} =====\" | tee -a logs/phase1_default.log"
    for clf in "${CLASSIFIERS[@]}"; do
      printf '%s\n' \
        "echo \"[${feat}] START: ${clf}  \$(date)\" | tee -a logs/phase1_default.log" \
        "\"\$PY\" -u ML.py --data ${DATA} --sample 0 --features ${feat} --modes \"${MODES}\" --only \"${clf}\" --resume --n_iter 10 >> logs/phase1_default.log 2>&1" \
        "rc=\$?" \
        "if [ \$rc -ne 0 ]; then echo \"[${feat}] FAILED: ${clf} (exit \$rc)  \$(date)\" | tee -a logs/phase1_default.log; fi" \
        "sync; sleep 1"
    done
  done
  cat <<'EOS'
echo "=== Phase1 done $(date) ===" | tee -a logs/phase1_default.log
"$PY" -u - <<'PY'
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
    summaries.append({
        "Feature": key,
        "Always Misclassified Count": len(df),
        "Test Size": test_n,
    })
write_master_misclassified(OUT_DIR / "always_misclassified_all_features.xlsx", frames, summaries)
print("Wrote", OUT_DIR / "always_misclassified_all_features.xlsx")
PY
EOS
} > "$SCRIPT"
chmod +x "$SCRIPT"

tmux new-session -d -s "$SESSION" -n run "bash $SCRIPT; echo DONE; exec bash"
echo "Started tmux session '$SESSION' running $SCRIPT"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $(pwd)/logs/phase1_default.log"
