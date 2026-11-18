"""python -m stacking_17 --size 4k --embedder tfidf"""
from __future__ import annotations

import argparse
import os

from .runner import build_master, run_one


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="17-suite bases + top meta: normal & cross-fold stacking"
    )
    p.add_argument("--size", choices=["4k", "67k"], default="4k")
    p.add_argument(
        "--embedder",
        choices=["tfidf", "minilm", "bert", "roberta", "t5"],
        default="tfidf",
    )
    p.add_argument("--top-k-meta", type=int, default=3)
    p.add_argument(
        "--ratios",
        default="0,0.1,0.25,0.4",
        help="Cross-fold exchange ratios (comma-separated)",
    )
    p.add_argument("--all", action="store_true", help="All sizes x embedders")
    p.add_argument("--master-only", action="store_true")
    args = p.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.master_only:
        build_master()
        return

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    jobs = []
    if args.all:
        for size in ("4k", "67k"):
            for emb in ("tfidf", "minilm", "bert", "roberta", "t5"):
                jobs.append((size, emb))
    else:
        jobs.append((args.size, args.embedder))

    from pathlib import Path

    for size, emb in jobs:
        marker = Path("results_stacking_17") / f"{size}_{emb}" / f"combined_{size}_{emb}.xlsx"
        if marker.exists() and not os.environ.get("FORCE"):
            print(f"[skip] {marker}")
            continue
        run_one(
            size,
            emb,
            top_k_meta=args.top_k_meta,
            exchange_ratios=ratios,
        )

    build_master()


if __name__ == "__main__":
    main()
