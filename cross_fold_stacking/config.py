"""Experiment configuration for cross-fold exchange stacking."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """All knobs for reproducibility and experiment variants."""

    random_seed: int = 42
    n_folds: int = 5
    holdout_fraction: float = 0.20
    exchange_ratio: float = 0.0  # 0.0, 0.10, 0.25, 0.40
    exchange_ratios: list[float] = field(default_factory=lambda: [0.0, 0.10, 0.25, 0.40])

    # Data
    data_csv: str = "/home/luk_viper/ML-Project/training_data.csv"
    sample_total: int = 4000  # 0 = full balanced minority; 4000 = 2k+2k
    embedder: str = "tfidf"  # tfidf | minilm
    svd_dim: int = 500

    # Models
    base_model: str = "logistic"  # logistic | linear_svc | extra_trees
    meta_model: str = "logistic"
    oof_folds: int = 5  # inner CV for OOF base predictions on shared train samples

    # Distribution tagging (for 75:25 composition reporting)
    # exchange_origin (primary) | claim_length_bin | char_quartile | semantic_category
    distribution_key: str = "exchange_origin"
    target_similar_pct: float = 0.75
    target_dissimilar_pct: float = 0.25

    # Paths
    output_dir: str = "results_cross_fold"
    experiment_name: str = "cross_fold_stacking"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


EXPERIMENT_LABELS = {
    0.0: "E0",
    0.10: "E1",
    0.25: "E2",
    0.40: "E3",
}
