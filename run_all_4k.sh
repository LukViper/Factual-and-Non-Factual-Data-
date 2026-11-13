#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PWD}/.venv/bin/python"
mkdir -p logs
for f in tfidf minilm bert roberta t5; do
  echo "===== balanced_4k / $f ====="
  "$PY" -u "balanced_4k/run_${f}.py" 2>&1 | tee -a "logs/balanced_4k_${f}.log"
done
echo "All 4k embedders finished."
