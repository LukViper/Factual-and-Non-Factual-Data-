#!/usr/bin/env python3
"""Durable one-classifier-at-a-time full tuning driver."""
from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYBIN = str(ROOT / ".venv" / "bin" / "python")
DATA = "/home/luk_viper/ML-Project/training_data.csv"
MODES = "Default,Grid Search,Random Search,Bayesian Optimization"
N_ITER = "10"
FEATURES = ["tfidf", "minilm", "bert", "roberta", "t5"]
SIZES = ["4k", "67k"]
CLASSIFIERS = [
    "Logistic Regression", "Perceptron", "LDA", "Naive Bayes", "Decision Tree", "SVM", "KNN",
    "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Extra Trees",
    "Random Forest", "Neural Network (MLP)", "Voting", "Stacking",
]
LOG = ROOT / "logs" / "full_tuning_resilient.log"
LOG.parent.mkdir(parents=True, exist_ok=True)


def say(msg: str) -> None:
    line = f"{msg}  {datetime.now()}\n"
    print(line, end="", flush=True)
    with LOG.open("a") as f:
        f.write(line)


def main() -> None:
    say("=== RESILIENT FULL TUNING start")
    for size in SIZES:
        for feat in FEATURES:
            for clf in CLASSIFIERS:
                say(f"[{size}/{feat}] START: {clf}")
                cmd = [
                    PYBIN, "-u", str(ROOT / "ML.py"),
                    "--data", DATA, "--size", size, "--features", feat,
                    "--modes", MODES, "--only", clf, "--n_iter", N_ITER, "--resume",
                ]
                if size == "67k":
                    cmd.append("--skip_cv")
                feat_log = ROOT / "logs" / f"full_{size}_{feat}.log"
                with feat_log.open("a") as fl:
                    rc = subprocess.call(cmd, stdout=fl, stderr=subprocess.STDOUT)
                say(f"[{size}/{feat}] {'DONE' if rc == 0 else f'FAILED({rc})'}: {clf}")
                time.sleep(1)
    say("Building final Excel workbooks")
    subprocess.call([PYBIN, "-u", str(ROOT / "ML.py"), "--build-final"])
    say("=== RESILIENT FULL TUNING done")


if __name__ == "__main__":
    main()
