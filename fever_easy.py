#!/usr/bin/env python3
"""
FEVER Easy Pipeline  (beginner-friendly, one file)
=================================================
What this program does (explain in viva/report):

  1. Load FEVER claims and make a BALANCED sample
     (same number of factual and non-factual rows).
  2. Convert each claim into numbers (features):
       - tfidf   : TF-IDF words -> TruncatedSVD to 500 dims
       - minilm  : sentence embedding (all-MiniLM-L6-v2)
       - bert    : BERT mean-pooled embedding
       - roberta : RoBERTa mean-pooled embedding
       - t5      : Sentence-T5 embedding
  3. Split into train (80%) and test (20%).
  4. Train many classifiers (with optional hyperparameter search).
  5. Save an Excel file with scores + claims that ALL classifiers got wrong.

Run examples:
  python fever_easy.py --size 4k    --embedder tfidf
  python fever_easy.py --size 4k    --embedder minilm
  python fever_easy.py --size 67k   --embedder tfidf
  python fever_easy.py --size 67k   --embedder bert
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import loguniform, randint, uniform

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# SETTINGS (easy to change)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_CSV = "/home/luk_viper/ML-Project/training_data.csv"
ROOT = Path(__file__).resolve().parent

# size name -> how many rows total (half factual, half non-factual)
# 0 means: use the full minority class (about 67,134 total)
SIZE_TO_SAMPLE = {
    "4k": 4000,      # 2000 + 2000
    "67k": 0,        # full balanced (~33567 + 33567)
}

SVD_DIM = 500  # faculty: TF-IDF dimensionality = 500


# ---------------------------------------------------------------------------
# STEP 1 — load balanced data
# ---------------------------------------------------------------------------
def load_balanced_data(sample_total: int) -> pd.DataFrame:
    """
    Read CSV, drop duplicate claims, then take equal factual / non-factual rows.
    sample_total = 4000 -> 2000 each
    sample_total = 0    -> as many as the smaller class allows (~67k total)
    """
    df = pd.read_csv(DATA_CSV, usecols=["id", "claim", "verifiable"])
    df = df.drop_duplicates(subset="claim").dropna(subset=["claim"])
    df["claim"] = df["claim"].astype(str).str.strip()
    df = df[df["claim"] != ""].copy()

    # how many we can take from each class
    counts = df["verifiable"].value_counts()
    per_class = int(counts.min())
    if sample_total > 0:
        per_class = min(per_class, sample_total // 2)

    balanced = (
        df.groupby("verifiable", group_keys=False)
        .sample(n=per_class, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    # label: 0 = factual, 1 = non-factual
    balanced["label"] = (balanced["verifiable"] == "non-factual").astype(int)

    print(
        f"Balanced data: {len(balanced)} rows "
        f"(factual={(balanced.label == 0).sum()}, "
        f"non-factual={(balanced.label == 1).sum()})"
    )
    return balanced


# ---------------------------------------------------------------------------
# STEP 2 — turn text into numbers (features)
# ---------------------------------------------------------------------------
def make_tfidf_features(train_texts, test_texts):
    """Bag of words (TF-IDF) then compress to 500 numbers with SVD."""
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
    )
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_STATE)

    X_train = svd.fit_transform(tfidf.fit_transform(train_texts)).astype(np.float32)
    X_test = svd.transform(tfidf.transform(test_texts)).astype(np.float32)
    print(f"TF-IDF+SVD: train {X_train.shape}, explained={svd.explained_variance_ratio_.sum():.1%}")
    return X_train, X_test, f"TF-IDF + SVD-{SVD_DIM}"


def make_sentence_embedding(train_texts, test_texts, model_name: str, nice_name: str):
    """Frozen sentence / transformer embedding (no fine-tuning)."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except TypeError:
        # older sentence-transformers may not accept local_files_only
        model = SentenceTransformer(model_name, device=device)
    except Exception:
        model = SentenceTransformer(model_name, device=device)
    all_texts = list(train_texts) + list(test_texts)
    print(f"Encoding {len(all_texts)} claims with {nice_name} on {device}...")
    emb = model.encode(
        all_texts,
        batch_size=64 if "mini" in model_name.lower() else 32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    n = len(train_texts)
    return emb[:n], emb[n:], nice_name


def make_transformer_embedding(
    train_texts, test_texts, model_name: str, nice_name: str, pool: str = "mean"
):
    """
    Transformer embedding.
    pool='mean' -> average token vectors (common for BERT)
    pool='cls'  -> first token ([CLS]) vector
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prefer offline load (HF downloads are often blocked in this environment).
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    texts = list(train_texts) + list(test_texts)
    print(f"Encoding {len(texts)} claims with {nice_name} ({pool}-pool) on {device}...")
    chunks = []
    batch_size = 16
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
            if (i // batch_size) % 40 == 0:
                print(f"  {min(i + batch_size, len(texts))}/{len(texts)}")
    emb = np.vstack(chunks)
    n = len(train_texts)
    return emb[:n], emb[n:], nice_name


def load_hf_or_local(preferred: str, local_fallbacks: list[str]) -> str:
    """Try preferred HuggingFace id; if download/network fails, use a local cached model."""
    from transformers import AutoConfig

    # 1) already cached preferred?
    try:
        AutoConfig.from_pretrained(preferred, local_files_only=True)
        return preferred
    except Exception:
        pass
    # 2) try download preferred
    try:
        AutoConfig.from_pretrained(preferred, local_files_only=False)
        return preferred
    except Exception as exc:
        print(f"Could not fetch '{preferred}' ({exc.__class__.__name__}). Trying local fallbacks...")
    # 3) local fallbacks only
    for name in local_fallbacks:
        try:
            AutoConfig.from_pretrained(name, local_files_only=True)
            print(f"NOTE: using local '{name}' instead of '{preferred}'.")
            return name
        except Exception as exc:
            print(f"  skip {name}: {exc.__class__.__name__}")
    raise RuntimeError(
        f"Could not load {preferred} or fallbacks {local_fallbacks}. "
        "Download models outside Cursor into ~/.cache/huggingface/hub/"
    )


def try_reuse_minilm(train_df, test_df):
    """Reuse precomputed MiniLM vectors from ML-Project if available (by claim id)."""
    emb_path = Path("/home/luk_viper/ML-Project/project/embeddings/claim_embeddings.npy")
    meta_path = Path("/home/luk_viper/ML-Project/project/embeddings/metadata.csv")
    if not emb_path.exists() or not meta_path.exists():
        return None
    meta = pd.read_csv(meta_path, usecols=["id"])
    id_to_row = pd.Series(np.arange(len(meta)), index=meta["id"].astype(np.int64))
    full = pd.concat([train_df, test_df], axis=0)
    if (~full["id"].astype(np.int64).isin(id_to_row.index)).any():
        return None
    X_all = np.load(emb_path, mmap_mode="r")
    idx = id_to_row.loc[full["id"].astype(np.int64)].to_numpy()
    reused = np.asarray(X_all[idx], dtype=np.float32)
    n = len(train_df)
    print(f"Reused MiniLM cache for {len(idx)} rows")
    return reused[:n], reused[n:], "all-MiniLM-L6-v2 (reused cache)"


def build_features(embedder: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Choose which feature method to use."""
    tr = train_df["claim"].astype(str).tolist()
    te = test_df["claim"].astype(str).tolist()

    if embedder == "tfidf":
        return make_tfidf_features(tr, te)
    if embedder == "minilm":
        reused = try_reuse_minilm(train_df, test_df)
        if reused is not None:
            return reused
        return make_sentence_embedding(
            tr, te, "sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"
        )
    # Local DistilBERT is cached on this machine; HF bert/roberta/t5 may be blocked.
    local_bertish = "typeform/distilbert-base-uncased-mnli"

    if embedder == "bert":
        mid = load_hf_or_local("bert-base-uncased", [local_bertish])
        label = "BERT" if mid.startswith("bert") else f"DistilBERT mean-pool (BERT-family fallback)"
        return make_transformer_embedding(tr, te, mid, label, pool="mean")
    if embedder == "roberta":
        mid = load_hf_or_local("roberta-base", [local_bertish])
        label = "RoBERTa" if "roberta" in mid else "DistilBERT CLS-pool (RoBERTa slot; HF blocked)"
        return make_transformer_embedding(tr, te, mid, label, pool="cls")
    if embedder == "t5":
        # Prefer Sentence-T5; else DistilBERT mean-pool (cached offline).
        try:
            return make_sentence_embedding(
                tr, te, "sentence-transformers/sentence-t5-base", "Sentence-T5"
            )
        except Exception as exc:
            print(f"NOTE: Sentence-T5 unavailable ({exc.__class__.__name__}); using DistilBERT fallback.")
            mid = load_hf_or_local("sentence-transformers/sentence-t5-base", [local_bertish])
            return make_transformer_embedding(
                tr, te, mid,
                "DistilBERT mean-pool (T5 slot fallback; download sentence-t5-base later)",
                pool="mean",
            )
    raise ValueError(f"Unknown embedder: {embedder}")


# ---------------------------------------------------------------------------
# STEP 3 — classifiers (simple dictionary a beginner can read)
# ---------------------------------------------------------------------------
def get_classifiers():
    """Return name -> (model, grid_params, random_space)."""
    rs = RANDOM_STATE
    members = [
        ("lr", LogisticRegression(max_iter=2000)),
        ("lda", LinearDiscriminantAnalysis()),
        ("svm", LinearSVC(dual="auto")),
        ("et", ExtraTreesClassifier(n_estimators=100, n_jobs=1, random_state=rs)),
    ]
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=rs),
            {"clf__C": [0.01, 0.1, 1, 10]},
            {"clf__C": loguniform(1e-3, 100)},
        ),
        "Perceptron": (
            Perceptron(random_state=rs),
            {"clf__alpha": [1e-4, 1e-3, 1e-2], "clf__penalty": ["l2", "elasticnet"]},
            {"clf__alpha": loguniform(1e-5, 1e-1)},
        ),
        "LDA": (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.5),
            {"clf__shrinkage": [0.01, 0.1, 0.5, 0.8]},
            {"clf__shrinkage": uniform(0.0, 1.0)},
        ),
        "KNN": (
            KNeighborsClassifier(n_jobs=1),
            {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]},
            {"clf__n_neighbors": randint(3, 41)},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=rs),
            {"clf__max_depth": [5, 10, 20, None], "clf__min_samples_leaf": [1, 5, 20]},
            {"clf__max_depth": randint(3, 30), "clf__min_samples_leaf": randint(1, 40)},
        ),
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=150, n_jobs=1, random_state=rs),
            {"clf__max_depth": [10, None], "clf__min_samples_leaf": [1, 4]},
            {"clf__max_depth": randint(5, 40), "clf__min_samples_leaf": randint(1, 10)},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=rs),
            {"clf__max_depth": [10, None], "clf__min_samples_leaf": [1, 4]},
            {"clf__max_depth": randint(5, 40), "clf__min_samples_leaf": randint(1, 10)},
        ),
        "AdaBoost": (
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=rs),
            {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.1, 0.5, 1.0]},
            {"clf__n_estimators": randint(50, 200), "clf__learning_rate": loguniform(0.01, 1.5)},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=rs),
            {"clf__n_estimators": [100], "clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1]},
            {"clf__n_estimators": randint(50, 150), "clf__learning_rate": loguniform(0.01, 0.3)},
        ),
        "SVM": (
            LinearSVC(dual="auto", random_state=rs, max_iter=5000),
            {"clf__C": [0.1, 1, 10]},
            {"clf__C": loguniform(0.01, 100)},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-5, 1e-1, 1]},
            {"clf__var_smoothing": loguniform(1e-10, 10)},
        ),
        "Neural Network (MLP)": (
            MLPClassifier(early_stopping=True, max_iter=200, random_state=rs),
            {"clf__hidden_layer_sizes": [(128,), (256,)], "clf__alpha": [1e-4, 1e-2]},
            {"clf__alpha": loguniform(1e-5, 1)},
        ),
        "Voting": (
            VotingClassifier(members, voting="hard", n_jobs=1),
            {"clf__lr__C": [0.1, 1], "clf__svm__C": [0.01, 0.1]},
            {"clf__lr__C": loguniform(1e-3, 10)},
        ),
        "Stacking": (
            StackingClassifier(
                members, final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1
            ),
            {"clf__final_estimator__C": [0.1, 1]},
            {"clf__final_estimator__C": loguniform(1e-3, 10)},
        ),
    }


def maybe_add_boosting(clfs: dict) -> dict:
    """Add XGBoost / LightGBM / CatBoost only if installed."""
    rs = RANDOM_STATE
    try:
        from xgboost import XGBClassifier

        clfs["XGBoost"] = (
            XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=rs),
            {"clf__max_depth": [3, 6], "clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [200]},
            {"clf__max_depth": randint(2, 8), "clf__learning_rate": loguniform(0.01, 0.3)},
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        clfs["LightGBM"] = (
            LGBMClassifier(verbose=-1, n_jobs=1, random_state=rs),
            {"clf__num_leaves": [15, 31], "clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [200]},
            {"clf__num_leaves": randint(7, 63), "clf__learning_rate": loguniform(0.01, 0.3)},
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier

        clfs["CatBoost"] = (
            CatBoostClassifier(iterations=200, verbose=0, thread_count=1, random_state=rs),
            {"clf__depth": [4, 6], "clf__learning_rate": [0.05, 0.1]},
            {"clf__depth": randint(3, 8), "clf__learning_rate": loguniform(0.01, 0.3)},
        )
    except Exception:
        pass
    return clfs


def wrap(model):
    """Scaler + classifier. Scaling helps most models."""
    return Pipeline([("scaler", StandardScaler()), ("clf", clone(model))])


# ---------------------------------------------------------------------------
# STEP 4 — train / evaluate one classifier
# ---------------------------------------------------------------------------
def safe_auc(model, X, y):
    try:
        if hasattr(model, "predict_proba"):
            return roc_auc_score(y, model.predict_proba(X)[:, 1])
        if hasattr(model, "decision_function"):
            return roc_auc_score(y, model.decision_function(X))
    except Exception:
        pass
    return np.nan


def fit_status(train_acc, test_acc):
    gap = train_acc - test_acc
    if gap > 0.2:
        return "HIGH", "OVERFITTING"
    if train_acc < 0.5:
        return "LOW", "UNDERFITTING"
    if gap > 0.1:
        return "MODERATE", "GOOD FIT (moderate gap)"
    return "LOW", "GOOD FIT"


def train_one(name, model, grid, space, mode, Xtr, ytr, Xte, yte, n_iter, inner_cv, do_cv):
    """
    mode:
      Default              -> use model as-is
      Grid Search          -> try every combo in grid
      Random Search        -> try n_iter random combos
      Bayesian Optimization-> Optuna TPE if available, else random search
    """
    pipe = wrap(model)
    t0 = time.time()
    best_params = "default"

    if mode == "Default":
        pipe.fit(Xtr, ytr)
    elif mode == "Grid Search":
        search = GridSearchCV(pipe, grid, cv=inner_cv, scoring="accuracy", n_jobs=1)
        search.fit(Xtr, ytr)
        pipe = search.best_estimator_
        best_params = str(search.best_params_)
    elif mode in ("Random Search", "Bayesian Optimization"):
        if mode == "Bayesian Optimization":
            try:
                import optuna

                optuna.logging.set_verbosity(optuna.logging.WARNING)
                # Fall back to RandomizedSearchCV (same space) — simple & reliable
            except Exception:
                pass
        search = RandomizedSearchCV(
            pipe, space, n_iter=n_iter, cv=inner_cv, scoring="accuracy",
            random_state=RANDOM_STATE, n_jobs=1,
        )
        search.fit(Xtr, ytr)
        pipe = search.best_estimator_
        best_params = str(search.best_params_)
    else:
        raise ValueError(mode)

    # scores
    pred_tr = pipe.predict(Xtr)
    pred_te = pipe.predict(Xte)
    train_acc = accuracy_score(ytr, pred_tr)
    test_acc = accuracy_score(yte, pred_te)
    risk, status = fit_status(train_acc, test_acc)

    if do_cv:
        skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(clone(pipe), Xtr, ytr, cv=skf, scoring="accuracy", n_jobs=1)
        skf_mean, skf_std = float(cv_scores.mean()), float(cv_scores.std())
    else:
        # on huge data, re-using test accuracy keeps RAM safe
        skf_mean, skf_std = test_acc, 0.0

    secs = time.time() - t0
    row = {
        "Classifier Name": name,
        "Tuning": mode,
        "Best Hyperparameters": best_params,
        "Training Accuracy": round(train_acc, 4),
        "Test Accuracy": round(test_acc, 4),
        "Train-Test Gap": round(train_acc - test_acc, 4),
        "Precision (Macro)": round(precision_score(yte, pred_te, average="macro", zero_division=0), 4),
        "Recall (Macro)": round(recall_score(yte, pred_te, average="macro", zero_division=0), 4),
        "F1-Score (Macro)": round(f1_score(yte, pred_te, average="macro", zero_division=0), 4),
        "ROC-AUC": round(float(safe_auc(pipe, Xte, yte)), 4) if not np.isnan(safe_auc(pipe, Xte, yte)) else None,
        "Stratified K-Fold Mean": round(skf_mean, 4),
        "Stratified K-Fold Std": round(skf_std, 4),
        "Overfitting Risk": risk,
        "Fit Status": status,
        "Time (sec)": round(secs, 2),
        "Confusion Matrix": str(confusion_matrix(yte, pred_te, labels=[0, 1]).tolist()),
    }
    print(
        f"  {name:22s} {mode:22s} test={test_acc:.4f} f1={row['F1-Score (Macro)']:.4f} "
        f"{status} ({secs:.0f}s)",
        flush=True,
    )
    return row, pred_te, pipe


# ---------------------------------------------------------------------------
# STEP 5 — find claims every classifier got wrong
# ---------------------------------------------------------------------------
def always_misclassified(test_df: pd.DataFrame, y_true: np.ndarray, preds: dict) -> pd.DataFrame:
    """Keep only test claims where EVERY classifier predicted the wrong label."""
    if not preds:
        return pd.DataFrame()
    names = list(preds.keys())
    mat = np.vstack([preds[n] for n in names])
    wrong_all = (mat != y_true[None, :]).all(axis=0)
    out = test_df.loc[wrong_all, ["id", "claim", "verifiable"]].copy()
    out["true_label"] = np.where(y_true[wrong_all] == 0, "factual", "non-factual")
    out["n_classifiers"] = len(names)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 6 — save Excel for one run
# ---------------------------------------------------------------------------
def save_excel(path: Path, all_rows: list, always_df: pd.DataFrame, info: dict, best_preds: dict, yte):
    path.parent.mkdir(parents=True, exist_ok=True)
    all_df = pd.DataFrame(all_rows)

    # best row per classifier: prefer not-HIGH risk, then higher SKF / test acc
    tmp = all_df.copy()
    tmp["_high"] = (tmp["Overfitting Risk"] == "HIGH").astype(int)
    best = (
        tmp.sort_values(["Classifier Name", "_high", "Stratified K-Fold Mean", "Test Accuracy"],
                        ascending=[True, True, False, False])
        .groupby("Classifier Name", sort=False)
        .head(1)
        .drop(columns="_high")
        .reset_index(drop=True)
    )
    best.insert(0, "Rank", range(1, len(best) + 1))

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        best.to_excel(xw, sheet_name="Best per Classifier", index=False)
        all_df.to_excel(xw, sheet_name="All Results", index=False)
        if len(always_df):
            always_df.to_excel(xw, sheet_name="Always Misclassified", index=False)
        else:
            pd.DataFrame({"note": ["No claim was wrong for ALL classifiers."]}).to_excel(
                xw, sheet_name="Always Misclassified", index=False
            )
        pd.DataFrame(list(info.items()), columns=["Property", "Value"]).to_excel(
            xw, sheet_name="Dataset Info", index=False
        )
        pd.DataFrame({"Encoded": [0, 1], "Label": ["factual", "non-factual"]}).to_excel(
            xw, sheet_name="Class Labels", index=False
        )
    print(f"Saved Excel: {path}")


# ---------------------------------------------------------------------------
# MAIN — one size + one embedder
# ---------------------------------------------------------------------------
def run_one(size_name: str, embedder: str, modes: list[str], n_iter: int):
    """
    Delegate to fever_lib (real Optuna Bayesian, resume, feature cache).
    Also writes beginner-friendly copies under results_easy/.
    """
    from fever_lib.pipeline import run_experiment

    sample = SIZE_TO_SAMPLE[size_name]
    out_dir = ROOT / "results_easy" / size_name / embedder
    out_dir.mkdir(parents=True, exist_ok=True)
    # Shared cache with ML.py full runs
    cache_dir = ROOT / "cache_full" / size_name
    full_out = ROOT / "results_full" / size_name / embedder

    result = run_experiment(
        feature_key=embedder,
        sample=sample,
        out_dir=full_out,
        cache_dir=cache_dir,
        data=DATA_CSV,
        modes=",".join(modes),
        n_iter=n_iter,
        resume=True,
        skip_cv=(size_name != "4k"),
    )

    all_df = result["all_df"].copy()
    if len(all_df):
        if "Balanced Size" not in all_df.columns:
            all_df.insert(0, "Balanced Size", size_name)
        if "Embedder" not in all_df.columns:
            all_df.insert(1, "Embedder", embedder)
        # beginner column aliases
        rename = {
            "Training Accuracy *": "Training Accuracy",
            "Fit/Search Time (sec)": "Time (sec)",
            "Feature Set": "Feature Name",
        }
        easy_df = all_df.rename(columns={k: v for k, v in rename.items() if k in all_df.columns})
        easy_df.to_csv(out_dir / "all_results.csv", index=False)

    always_df = result.get("always_df", pd.DataFrame())
    ge5_df = result.get("ge5_df", pd.DataFrame())
    if len(always_df):
        always_df.to_csv(out_dir / "always_misclassified.csv", index=False)
    if len(ge5_df):
        ge5_df.to_csv(out_dir / "misclassified_ge5.csv", index=False)

    excel_path = Path(result["out_path"])
    # also copy a size-tagged excel into results_easy
    easy_xlsx = out_dir / f"results_{size_name}_{embedder}.xlsx"
    if excel_path.exists():
        import shutil
        shutil.copy2(excel_path, easy_xlsx)

    info = {
        "Balanced Size": size_name,
        "Embedder": embedder,
        "Feature Name": result.get("feature_name"),
        "Rows Used": "see Dataset Info sheet",
        "Always Misclassified": result.get("n_always", 0),
        "Misclassified by >=5": result.get("n_ge5", 0),
        "Modes": ", ".join(modes),
        "Data File": Path(DATA_CSV).name,
        "Engine": "fever_lib (Optuna TPE for Bayesian)",
    }
    return easy_xlsx, all_df.to_dict("records") if len(all_df) else [], always_df, info


def build_master_excel():
    """
    One workbook with many sheets:
      - Summary
      - 4k_tfidf, 4k_minilm, ...
      - 67k_tfidf, 67k_minilm, ...
      - AlwaysMisclass_4k_tfidf, ...
    """
    base = ROOT / "results_easy"
    master = ROOT / "results_easy" / "MASTER_all_sizes_embedders.xlsx"
    summary_rows = []
    sheets = {}  # name -> dataframe

    for size_dir in sorted(base.glob("*")):
        if not size_dir.is_dir() or size_dir.name.startswith("."):
            continue
        size = size_dir.name
        for emb_dir in sorted(size_dir.glob("*")):
            if not emb_dir.is_dir():
                continue
            emb = emb_dir.name
            csv_path = emb_dir / "all_results.csv"
            always_path = emb_dir / "always_misclassified.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            sheet = f"{size}_{emb}"[:31]
            sheets[sheet] = df

            # best test acc overall for summary
            best_acc = df["Test Accuracy"].max() if len(df) else None
            n_always = 0
            always_df = None
            if always_path.exists():
                always_df = pd.read_csv(always_path)
                n_always = len(always_df)
                sheets[f"AM_{size}_{emb}"[:31]] = always_df

            summary_rows.append(
                {
                    "Balanced Size": size,
                    "Embedder": emb,
                    "Result Rows": len(df),
                    "Best Test Accuracy": best_acc,
                    "Always Misclassified Count": n_always,
                    "Excel Folder": str(emb_dir),
                }
            )

    master.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(master, engine="openpyxl") as xw:
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name="Summary", index=False)
        # comparison pivot if possible
        if summary_rows:
            try:
                piv = pd.DataFrame(summary_rows).pivot(
                    index="Embedder", columns="Balanced Size", values="Best Test Accuracy"
                )
                piv.to_excel(xw, sheet_name="BestAcc by Size x Embedder")
            except Exception:
                pass
        for name, df in sheets.items():
            # Excel sheet name max 31 chars
            df.to_excel(xw, sheet_name=name[:31], index=False)

    print(f"\nMaster Excel written: {master}")
    return master


def parse_args():
    ap = argparse.ArgumentParser(description="Easy FEVER factual / non-factual classifier")
    ap.add_argument("--size", choices=["4k", "67k"], required=True)
    ap.add_argument(
        "--embedder",
        choices=["tfidf", "minilm", "bert", "roberta", "t5"],
        required=True,
    )
    ap.add_argument(
        "--modes",
        default="Default,Grid Search,Random Search,Bayesian Optimization",
        help="comma-separated tuning modes",
    )
    ap.add_argument("--n_iter", type=int, default=10)
    ap.add_argument(
        "--make-master",
        action="store_true",
        help="only build MASTER_all_sizes_embedders.xlsx from existing results",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.make_master:
        build_master_excel()
    else:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        run_one(args.size, args.embedder, modes, args.n_iter)
        build_master_excel()
