#!/usr/bin/env bash
# Run every balanced-size x embedder, then refresh master Excel.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PWD}/.venv/bin/python"
mkdir -p logs

MODES_4K="${MODES_4K:-Default,Grid Search,Random Search,Bayesian Optimization}"
MODES_67K="${MODES_67K:-Default,Grid Search,Random Search,Bayesian Optimization}"

for emb in tfidf minilm bert roberta t5; do
  echo "===== 4k / $emb =====" | tee -a logs/easy_run.log
  "$PY" -u fever_easy.py --size 4k --embedder "$emb" --modes "$MODES_4K" \
    2>&1 | tee -a "logs/easy_4k_${emb}.log"
done

for emb in tfidf minilm bert roberta t5; do
  echo "===== 67k / $emb =====" | tee -a logs/easy_run.log
  "$PY" -u fever_easy.py --size 67k --embedder "$emb" --modes "$MODES_67K" \
    2>&1 | tee -a "logs/easy_67k_${emb}.log"
done

"$PY" -u fever_easy.py --size 4k --embedder tfidf --make-master
echo "ALL DONE. Master Excel: results_easy/MASTER_all_sizes_embedders.xlsx"
