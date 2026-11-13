#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PWD}/.venv/bin/python"
mkdir -p logs
for f in tfidf minilm bert roberta t5; do
  echo "===== balanced_67134 / $f ====="
  "$PY" -u "balanced_67134/run_${f}.py" 2>&1 | tee -a "logs/balanced_67134_${f}.log"
done
echo "All 67k embedders finished."
