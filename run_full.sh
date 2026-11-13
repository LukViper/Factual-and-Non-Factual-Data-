#!/usr/bin/env bash
# Full faculty experiment: TF-IDF@500 + MiniLM/BERT/RoBERTa/T5, all classifiers,
# balanced FEVER >50k, always-misclassified export.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PWD}/.venv/bin/python"
mkdir -p logs outputs cache

echo "[1/2] Default mode (all features) — builds misclassified lists"
"$PY" -u ML.py \
  --data /home/luk_viper/ML-Project/training_data.csv \
  --sample 0 \
  --features tfidf,minilm,bert,roberta,t5 \
  --modes Default \
  --n_iter 10 \
  | tee logs/phase1_default.log

echo "[2/2] Full tuning (Grid / Random / Bayesian) on all features"
"$PY" -u ML.py \
  --data /home/luk_viper/ML-Project/training_data.csv \
  --sample 0 \
  --features tfidf,minilm,bert,roberta,t5 \
  --modes "Grid Search,Random Search,Bayesian Optimization" \
  --n_iter 10 \
  | tee logs/phase2_tuning.log

echo "Done. See outputs/"
