"""Dataset loading, balancing, holdout split, and feature building."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
EMB_CACHE = ROOT / "cache" / "cross_fold_embeddings"
LOCAL_BERTISH = "typeform/distilbert-base-uncased-mnli"
MINILM_EMB = Path("/home/luk_viper/ML-Project/project/embeddings/claim_embeddings.npy")
MINILM_META = Path("/home/luk_viper/ML-Project/project/embeddings/metadata.csv")

SUPPORTED_EMBEDDERS = ("tfidf", "minilm", "bert", "roberta", "t5")


@dataclass
class DatasetBundle:
    """Full dataset after balancing + 80/20 holdout; features fit on development only."""

    development: pd.DataFrame
    holdout: pd.DataFrame
    X_dev: np.ndarray
    X_holdout: np.ndarray
    feature_name: str
    sample_ids: np.ndarray


def load_balanced_frame(data_csv: str, sample_total: int, random_seed: int) -> pd.DataFrame:
    """Equal factual / non-factual claims (same rules as fever_easy)."""
    df = pd.read_csv(data_csv, usecols=["id", "claim", "verifiable"])
    df = df.drop_duplicates(subset="claim").dropna(subset=["claim"])
    df["claim"] = df["claim"].astype(str).str.strip()
    df = df[df["claim"] != ""].copy()

    per_class = int(df["verifiable"].value_counts().min())
    if sample_total > 0:
        per_class = min(per_class, sample_total // 2)

    balanced = (
        df.groupby("verifiable", group_keys=False)
        .sample(n=per_class, random_state=random_seed)
        .reset_index(drop=True)
    )
    balanced["label"] = (balanced["verifiable"] == "non-factual").astype(int)
    balanced["sample_uid"] = np.arange(len(balanced), dtype=np.int64)
    return balanced


def split_development_holdout(
    df: pd.DataFrame,
    holdout_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80% development / 20% untouched holdout (stratified)."""
    dev, hold = train_test_split(
        df,
        test_size=holdout_fraction,
        random_state=random_seed,
        stratify=df["label"],
    )
    return dev.reset_index(drop=True), hold.reset_index(drop=True)


def _scale_split(X_all: np.ndarray, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[:n_train]).astype(np.float32)
    X_te = scaler.transform(X_all[n_train:]).astype(np.float32)
    return X_tr, X_te


def _build_tfidf(
    train_texts: list[str],
    test_texts: list[str],
    svd_dim: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
    )
    svd = TruncatedSVD(n_components=svd_dim, random_state=random_seed)
    X_tr = svd.fit_transform(tfidf.fit_transform(train_texts)).astype(np.float32)
    X_te = svd.transform(tfidf.transform(test_texts)).astype(np.float32)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    return X_tr, X_te, f"TF-IDF + SVD-{svd_dim}"


def _reuse_minilm_project_cache(train_df: pd.DataFrame, test_df: pd.DataFrame):
    if not MINILM_EMB.exists() or not MINILM_META.exists():
        return None
    meta = pd.read_csv(MINILM_META, usecols=["id"])
    id_to_row = pd.Series(np.arange(len(meta)), index=meta["id"].astype(np.int64))
    full = pd.concat([train_df, test_df], axis=0)
    if not full["id"].astype(np.int64).isin(id_to_row.index).all():
        return None
    X_all = np.load(MINILM_EMB, mmap_mode="r")
    idx = id_to_row.loc[full["id"].astype(np.int64)].to_numpy()
    reused = np.asarray(X_all[idx], dtype=np.float32)
    n = len(train_df)
    X_tr, X_te = _scale_split(reused, n)
    return X_tr, X_te, "all-MiniLM-L6-v2 (reused cache)"


def _encode_sentence_transformer(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except TypeError:
        model = SentenceTransformer(model_name, device=device)
    except Exception:
        model = SentenceTransformer(model_name, device=device)
    return model.encode(
        texts,
        batch_size=64 if "mini" in model_name.lower() else 32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def _load_hf_or_local(preferred: str, local_fallbacks: list[str]) -> str:
    """Prefer local cache; skip network if HF is blocked (common in this env)."""
    import os
    from transformers import AutoConfig

    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE", "0"
    ) == "1"

    try:
        AutoConfig.from_pretrained(preferred, local_files_only=True)
        return preferred
    except Exception:
        pass

    if not offline:
        try:
            AutoConfig.from_pretrained(preferred, local_files_only=False)
            return preferred
        except Exception as exc:
            print(
                f"Could not fetch '{preferred}' ({exc.__class__.__name__}). "
                "Trying local fallbacks..."
            )
    else:
        print(f"Offline mode: skip download of '{preferred}'; use local fallbacks.")

    for name in local_fallbacks:
        try:
            AutoConfig.from_pretrained(name, local_files_only=True)
            print(f"NOTE: using local '{name}' instead of '{preferred}'.")
            return name
        except Exception as exc:
            print(f"  skip {name}: {exc.__class__.__name__}")
    raise RuntimeError(f"Could not load {preferred} or fallbacks {local_fallbacks}")


def _encode_transformer(
    texts: list[str], model_name: str, pool: str = "mean"
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    chunks = []
    batch_size = 16
    print(f"Encoding {len(texts)} claims with {model_name} ({pool}-pool) on {device}...")
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            if pool == "cls":
                pooled = hidden[:, 0, :]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
            if (i // batch_size) % 50 == 0:
                print(f"  {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(chunks)


def _cache_paths(embedder: str, n: int, random_seed: int) -> tuple[Path, Path]:
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    stem = f"{embedder}_n{n}_rs{random_seed}"
    return EMB_CACHE / f"{stem}.npy", EMB_CACHE / f"{stem}_ids.npy"


def _load_or_build_neural_embeddings(
    embedder: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Encode development+holdout; cache by balanced size + seed + embedder."""
    full = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    n_all = len(full)
    cache_x, cache_ids = _cache_paths(embedder, n_all, random_seed)

    if cache_x.exists() and cache_ids.exists():
        ids = np.load(cache_ids)
        if np.array_equal(ids, full["id"].astype(np.int64).to_numpy()):
            emb = np.load(cache_x)
            print(f"Reused embedding cache {cache_x.name} shape={emb.shape}")
            X_tr, X_te = _scale_split(emb, len(train_df))
            return X_tr, X_te, f"{embedder} (disk cache)"

    texts = full["claim"].astype(str).tolist()
    ids = full["id"].astype(np.int64).to_numpy()

    def _save(emb_arr: np.ndarray, label: str, also: list[str] | None = None):
        np.save(cache_x, emb_arr)
        np.save(cache_ids, ids)
        print(f"Saved embedding cache {cache_x}")
        for alt in also or []:
            ax, ai = _cache_paths(alt, n_all, random_seed)
            if not ax.exists():
                np.save(ax, emb_arr)
                np.save(ai, ids)
                print(f"Also cached as {ax.name} (shared fallback)")
        X_tr, X_te = _scale_split(emb_arr, len(train_df))
        return X_tr, X_te, label

    if embedder == "minilm":
        reused = _reuse_minilm_project_cache(train_df, test_df)
        if reused is not None:
            if MINILM_EMB.exists() and MINILM_META.exists():
                meta = pd.read_csv(MINILM_META, usecols=["id"])
                id_to_row = pd.Series(np.arange(len(meta)), index=meta["id"].astype(np.int64))
                idx = id_to_row.loc[ids].to_numpy()
                raw = np.asarray(np.load(MINILM_EMB, mmap_mode="r")[idx], dtype=np.float32)
                np.save(cache_x, raw)
                np.save(cache_ids, ids)
            return reused
        emb = _encode_sentence_transformer(texts, "sentence-transformers/all-MiniLM-L6-v2")
        return _save(emb, "all-MiniLM-L6-v2")

    if embedder == "bert":
        mid = _load_hf_or_local("bert-base-uncased", [LOCAL_BERTISH])
        if mid.startswith("bert"):
            emb = _encode_transformer(texts, mid, pool="mean")
            return _save(emb, "BERT")
        # DistilBERT mean — also usable as T5 fallback
        emb = _encode_transformer(texts, mid, pool="mean")
        return _save(emb, "DistilBERT mean-pool (BERT fallback)", also=["t5"])

    if embedder == "roberta":
        mid = _load_hf_or_local("roberta-base", [LOCAL_BERTISH])
        if "roberta" in mid:
            emb = _encode_transformer(texts, mid, pool="cls")
            return _save(emb, "RoBERTa")
        emb = _encode_transformer(texts, mid, pool="cls")
        return _save(emb, "DistilBERT CLS-pool (RoBERTa fallback)")

    if embedder == "t5":
        # Prefer existing bert DistilBERT-mean cache (same vectors when HF blocked)
        bert_x, bert_ids = _cache_paths("bert", n_all, random_seed)
        if bert_x.exists() and bert_ids.exists() and np.array_equal(np.load(bert_ids), ids):
            emb = np.load(bert_x)
            print(f"Reused BERT/DistilBERT-mean cache for T5 slot: {bert_x.name}")
            np.save(cache_x, emb)
            np.save(cache_ids, ids)
            X_tr, X_te = _scale_split(emb, len(train_df))
            return X_tr, X_te, "DistilBERT mean-pool (T5 slot; shared with BERT fallback)"
        try:
            emb = _encode_sentence_transformer(texts, "sentence-transformers/sentence-t5-base")
            return _save(emb, "Sentence-T5")
        except Exception as exc:
            print(f"NOTE: Sentence-T5 unavailable ({exc.__class__.__name__}); DistilBERT fallback.")
            mid = _load_hf_or_local("sentence-transformers/sentence-t5-base", [LOCAL_BERTISH])
            emb = _encode_transformer(texts, mid, pool="mean")
            return _save(emb, "DistilBERT mean-pool (T5 fallback)", also=["bert"])

    raise ValueError(f"Unknown neural embedder: {embedder}")


def prepare_dataset(
    data_csv: str,
    sample_total: int,
    holdout_fraction: float,
    random_seed: int,
    embedder: str = "tfidf",
    svd_dim: int = 500,
) -> DatasetBundle:
    """Load, balance, split holdout, build features (scaler/SVD fit on development only)."""
    embedder = embedder.lower().strip()
    if embedder not in SUPPORTED_EMBEDDERS:
        raise ValueError(f"Unsupported embedder: {embedder}. Choose from {SUPPORTED_EMBEDDERS}")

    balanced = load_balanced_frame(data_csv, sample_total, random_seed)
    development, holdout = split_development_holdout(balanced, holdout_fraction, random_seed)

    if embedder == "tfidf":
        X_dev, X_holdout, fname = _build_tfidf(
            development["claim"].astype(str).tolist(),
            holdout["claim"].astype(str).tolist(),
            svd_dim,
            random_seed,
        )
    else:
        X_dev, X_holdout, fname = _load_or_build_neural_embeddings(
            embedder, development, holdout, random_seed
        )

    return DatasetBundle(
        development=development,
        holdout=holdout,
        X_dev=X_dev,
        X_holdout=X_holdout,
        feature_name=fname,
        sample_ids=development["sample_uid"].to_numpy(),
    )
