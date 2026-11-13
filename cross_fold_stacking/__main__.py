"""python -m cross_fold_stacking ..."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .runner import run_experiments


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-fold data-exchange stacking experiments (E0–E3)"
    )
    p.add_argument("--config", type=str, default="", help="JSON config path")
    p.add_argument("--size", choices=["4k", "67k"], default="4k")
    p.add_argument(
        "--embedder",
        choices=["tfidf", "minilm", "bert", "roberta", "t5"],
        default="tfidf",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ratios",
        type=str,
        default="0,0.1,0.25,0.4",
        help="Comma-separated exchange ratios",
    )
    p.add_argument("--base-model", default="logistic")
    p.add_argument("--meta-model", default="logistic")
    p.add_argument("--distribution-key", default="exchange_origin")
    p.add_argument("--output-dir", default="results_cross_fold")
    p.add_argument("--name", default="cross_fold_stacking")
    p.add_argument(
        "--data-csv",
        default="/home/luk_viper/ML-Project/training_data.csv",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.config:
        cfg = ExperimentConfig.load(args.config)
    else:
        sample_total = 4000 if args.size == "4k" else 0
        ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
        cfg = ExperimentConfig(
            random_seed=args.seed,
            sample_total=sample_total,
            embedder=args.embedder,
            exchange_ratios=ratios,
            base_model=args.base_model,
            meta_model=args.meta_model,
            distribution_key=args.distribution_key,
            output_dir=args.output_dir,
            experiment_name=args.name,
            data_csv=args.data_csv,
        )
    run_experiments(cfg)


if __name__ == "__main__":
    main()
