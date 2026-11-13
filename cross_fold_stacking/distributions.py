"""
Distribution tagging and training-composition reporting.

Primary composition (matches the exchange research idea)
--------------------------------------------------------
For each model, training samples are labeled:

  similar     = still in this model's train and originally allocated as its train
                (not swapped in from the other model's exclusive fold)
  dissimilar  = exchanged *into* this model's train from the other exclusive fold
                (B for Base, A for Meta)

With equal folds of size s and exchange ratio r (|A|=|B|=r·s):
  similar_pct = 1 − r/4 , dissimilar_pct = r/4
  → 75:25 appears only at r=1.0 (full F1↔F5 swap), not at E0–E3.
  We never rewrite the allocation to force 75:25; we report the actual mix.

Optional domain keys (claim_length_bin, semantic_category, …) are reported
as a *secondary* breakdown and must not silently redefine the exchange.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exchange import ExchangeResult


@dataclass
class CompositionReport:
    model: str  # "base" | "meta"
    n_train: int
    n_test: int
    similar_count: int
    dissimilar_count: int
    similar_pct: float
    dissimilar_pct: float
    target_similar_pct: float
    target_dissimilar_pct: float
    target_achieved: bool
    distribution_key: str
    notes: str
    # Optional domain breakdown among train samples
    domain_majority_pct: float | None = None


def assign_distribution_labels(
    development: pd.DataFrame,
    key: str = "exchange_origin",
    n_bins: int = 4,
) -> np.ndarray:
    """
    Optional domain / pseudo-domain labels (orthogonal to exchange roles).

    Keys:
      - exchange_origin / none: placeholder zeros (composition uses exchange sets)
      - claim_length_bin / char_quartile: length quartiles
      - semantic_category: coarse keyword buckets
    """
    n = len(development)
    if key in ("none", "", None, "exchange_origin"):
        return np.zeros(n, dtype=int)

    if key in ("claim_length_bin", "char_quartile"):
        lengths = development["claim"].astype(str).str.len().to_numpy()
        order = lengths.argsort(kind="mergesort")
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        return np.minimum(ranks * n_bins // max(n, 1), n_bins - 1)

    if key == "semantic_category":
        claims = development["claim"].astype(str).str.lower()
        science = claims.str.contains(
            r"\b(scientist|physics|biology|chemistry|nasa|atom|dna|planet)\b",
            regex=True,
        )
        politics = claims.str.contains(
            r"\b(president|election|parliament|minister|senator|government)\b",
            regex=True,
        )
        labels = np.full(n, 2, dtype=int)
        labels[science.to_numpy()] = 0
        labels[politics.to_numpy()] = 1
        return labels

    raise ValueError(f"Unknown distribution_key: {key}")


def _composition_for_model(
    model: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dist_labels: np.ndarray,
    exchange: ExchangeResult,
    target_similar: float,
    target_dissimilar: float,
    distribution_key: str,
) -> CompositionReport:
    if model == "base":
        dissimilar_set = set(exchange.subset_b.tolist())
    else:
        dissimilar_set = set(exchange.subset_a.tolist())

    dissimilar = sum(1 for i in train_idx if int(i) in dissimilar_set)
    similar = len(train_idx) - dissimilar
    n_train = len(train_idx)
    sim_pct = similar / n_train if n_train else 0.0
    dis_pct = dissimilar / n_train if n_train else 0.0
    achieved = (
        abs(sim_pct - target_similar) <= 0.05
        and abs(dis_pct - target_dissimilar) <= 0.05
    )

    note_parts = []
    if not achieved:
        note_parts.append(
            f"Target {target_similar:.0%}:{target_dissimilar:.0%} not achieved "
            f"with exchange_ratio={exchange.exchange_ratio:.0%} "
            f"(exchange-origin mix is 1−r/4 : r/4); reporting actual."
        )

    domain_maj = None
    if distribution_key not in ("none", "", None, "exchange_origin") and n_train:
        labs = dist_labels[train_idx]
        vals, counts = np.unique(labs, return_counts=True)
        domain_maj = float(counts.max() / n_train)
        note_parts.append(
            f"Secondary domain '{distribution_key}': majority bucket "
            f"{domain_maj:.1%} of train (not used to rewrite exchange)."
        )

    return CompositionReport(
        model=model,
        n_train=n_train,
        n_test=len(test_idx),
        similar_count=similar,
        dissimilar_count=dissimilar,
        similar_pct=sim_pct,
        dissimilar_pct=dis_pct,
        target_similar_pct=target_similar,
        target_dissimilar_pct=target_dissimilar,
        target_achieved=achieved,
        distribution_key="exchange_origin",
        notes=" ".join(note_parts),
        domain_majority_pct=domain_maj,
    )


def report_compositions(
    exchange: ExchangeResult,
    dist_labels: np.ndarray,
    distribution_key: str,
    target_similar: float = 0.75,
    target_dissimilar: float = 0.25,
) -> dict[str, CompositionReport]:
    return {
        "base": _composition_for_model(
            "base",
            exchange.base_train_idx,
            exchange.base_test_idx,
            dist_labels,
            exchange,
            target_similar,
            target_dissimilar,
            distribution_key,
        ),
        "meta": _composition_for_model(
            "meta",
            exchange.meta_train_idx,
            exchange.meta_test_idx,
            dist_labels,
            exchange,
            target_similar,
            target_dissimilar,
            distribution_key,
        ),
    }


def composition_to_frame(reports: dict[str, CompositionReport]) -> pd.DataFrame:
    rows = []
    for r in reports.values():
        rows.append(
            {
                "model": r.model,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "similar_count": r.similar_count,
                "dissimilar_count": r.dissimilar_count,
                "similar_pct": round(r.similar_pct, 4),
                "dissimilar_pct": round(r.dissimilar_pct, 4),
                "target_similar_pct": r.target_similar_pct,
                "target_dissimilar_pct": r.target_dissimilar_pct,
                "target_achieved": r.target_achieved,
                "distribution_key": r.distribution_key,
                "domain_majority_pct": r.domain_majority_pct,
                "notes": r.notes,
            }
        )
    return pd.DataFrame(rows)
