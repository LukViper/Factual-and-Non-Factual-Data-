#!/usr/bin/env python3
"""
FEVER balanced classification CLI (real Optuna Bayesian + 4 tuning modes).

Examples:
  python ML.py --size 4k  --features tfidf,minilm --modes Default,Grid Search,Random Search,Bayesian Optimization
  python ML.py --size 67k --features tfidf --modes Default --skip_cv
  python ML.py --build-final
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fever_lib.pipeline import CLASSIFIERS, DEFAULT_DATA, EMBEDDERS, run_experiment

ROOT = Path(__file__).resolve().parent
MODES_ALL = "Default,Grid Search,Random Search,Bayesian Optimization"


def parse_args():
    ap = argparse.ArgumentParser(description="FEVER factual vs non-factual — full tuning")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--size", choices=["4k", "67k"], default="4k",
                    help="balanced size: 4k (2k+2k) or 67k (full minority class)")
    ap.add_argument("--sample", type=int, default=None,
                    help="override sample size (0 = full balanced ~67k)")
    ap.add_argument("--features", default="tfidf,minilm,bert,roberta,t5")
    ap.add_argument("--modes", default=MODES_ALL)
    ap.add_argument("--n_iter", type=int, default=10)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated classifier names")
    ap.add_argument("--out-root", default=str(ROOT / "results_full"))
    ap.add_argument("--cache-root", default=str(ROOT / "cache_full"))
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip_cv", action="store_true",
                    help="skip outer 5-fold (recommended for 67k)")
    ap.add_argument("--recompute_features", action="store_true")
    ap.add_argument("--build-final", action="store_true",
                    help="only merge existing results into final Excel workbooks")
    ap.add_argument("--min-wrong", type=int, default=5,
                    help="threshold for majority-misclassified Excel")
    return ap.parse_args()


def size_to_sample(size: str) -> int:
    return 4000 if size == "4k" else 0


def build_final(out_root: Path, min_wrong: int = 5):
    from build_final_outputs import build_all
    return build_all(out_root, min_wrong=min_wrong)


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    if args.build_final:
        paths = build_final(out_root, min_wrong=args.min_wrong)
        print("Final workbooks:", paths)
        return

    sample = args.sample if args.sample is not None else size_to_sample(args.size)
    size_name = "4k" if sample == 4000 else ("67k" if sample == 0 else f"n{sample}")
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    resume = not args.no_resume
    skip_cv = args.skip_cv or sample == 0 or (sample is not None and sample >= 20000)

    for feat in features:
        if feat != "tfidf" and feat not in EMBEDDERS:
            raise SystemExit(f"Unknown feature '{feat}'. Use tfidf or {list(EMBEDDERS)}")
        out_dir = out_root / size_name / feat
        cache_dir = Path(args.cache_root) / size_name
        print(f"\n===== {size_name} / {feat} | modes={args.modes} =====", flush=True)
        run_experiment(
            feature_key=feat,
            sample=sample,
            out_dir=out_dir,
            cache_dir=cache_dir,
            data=args.data,
            modes=args.modes,
            n_iter=args.n_iter,
            inner=args.inner,
            only=args.only,
            resume=resume,
            skip_cv=skip_cv,
            recompute_features=args.recompute_features,
        )

    paths = build_final(out_root, min_wrong=args.min_wrong)
    print("\nDone. Final workbooks:", paths)


if __name__ == "__main__":
    main()
