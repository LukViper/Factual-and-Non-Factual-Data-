"""
Shared FEVER classification engine used by balanced_4k/ and balanced_67134/ runners.
"""
from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RS = 42
GAP_TARGET = 0.15
MAX_SVD = 500
LABELS = ["factual", "non-factual"]
DEFAULT_DATA = "/home/luk_viper/ML-Project/training_data.csv"
N_JOBS = 1
ROOT = Path(__file__).resolve().parent.parent

# HuggingFace / SentenceTransformers model ids
EMBEDDERS = {
    "minilm": {
        "label": "all-MiniLM-L6-v2",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "kind": "sentence_transformer",
        "dim": 384,
        # Reuse full-corpus MiniLM cache from the sibling ML-Project if present
        "reuse_project_cache": Path("/home/luk_viper/ML-Project/project/embeddings"),
    },
    "bert": {
        "label": "BERT (bert-base-uncased)",
        "model": "bert-base-uncased",
        "kind": "mean_pool_transformer",
        "dim": 768,
    },
    "roberta": {
        "label": "RoBERTa (roberta-base)",
        "model": "roberta-base",
        "kind": "mean_pool_transformer",
        "dim": 768,
    },
    "t5": {
        "label": "Sentence-T5 (sentence-t5-base)",
        "model": "sentence-transformers/sentence-t5-base",
        "kind": "sentence_transformer",
        "dim": 768,
    },
}


# ---------------------------------------------------------------- features
class FirstK(BaseEstimator, TransformerMixin):
    """Keeps the first k SVD components (ordered by explained variance)."""
    def __init__(self, n_components=MAX_SVD):
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        k = int(self.n_components)
        return X[:, :k]


def make_pipe(clf, use_svd_k: bool):
    steps = []
    if use_svd_k:
        steps.append(("svd", FirstK()))
    steps += [("scaler", StandardScaler()), ("clf", clone(clf))]
    return Pipeline(steps)


# ---------------------------------------------------------------- classifiers
def ensemble_members():
    return [("lr", LogisticRegression(max_iter=2000)),
            ("lda", LinearDiscriminantAnalysis()),
            ("svm", LinearSVC(dual="auto")),
            ("et", ExtraTreesClassifier(n_estimators=200, n_jobs=1, random_state=RS))]


K_GRID = [250, 500]
K_SPACE = ("int", 100, 500)

CLASSIFIERS = {
    "Logistic Regression": ("Linear", LogisticRegression(max_iter=2000, random_state=RS),
        {"C": [0.001, 0.01, 0.1, 1, 10]},
        {"C": ("log", 1e-4, 100)}),
    "Perceptron": ("Linear", Perceptron(random_state=RS),
        {"alpha": [1e-4, 1e-3, 1e-2], "penalty": ["l2", "elasticnet"], "eta0": [0.1, 1.0]},
        {"alpha": ("log", 1e-5, 1e-1), "penalty": ("cat", ["l2", "l1", "elasticnet"]), "eta0": ("log", 1e-3, 1)}),
    "LDA": ("Linear (Generative)", LinearDiscriminantAnalysis(solver="lsqr"),
        {"shrinkage": [0.01, 0.1, 0.3, 0.5, 0.8]},
        {"shrinkage": ("float", 0.0, 1.0)}),
    "KNN": ("Distance-Based", KNeighborsClassifier(n_jobs=1),
        {"n_neighbors": [5, 11, 21, 31], "weights": ["uniform", "distance"], "metric": ["euclidean", "cosine"]},
        {"n_neighbors": ("int", 3, 51), "weights": ("cat", ["uniform", "distance"]), "metric": ("cat", ["euclidean", "cosine"])}),
    "Decision Tree": ("Tree-Based", DecisionTreeClassifier(random_state=RS),
        {"max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 5, 20]},
        {"max_depth": ("int", 3, 30), "min_samples_leaf": ("int", 1, 50),
         "criterion": ("cat", ["gini", "entropy"]), "ccp_alpha": ("float", 0.0, 0.01)}),
    "Extra Trees": ("Tree-Based (Ensemble)", ExtraTreesClassifier(n_jobs=1, random_state=RS),
        {"n_estimators": [200], "max_depth": [10, None], "min_samples_leaf": [1, 4]},
        {"n_estimators": ("int", 100, 300), "max_depth": ("int", 5, 40), "min_samples_leaf": ("int", 1, 10),
         "max_features": ("cat", ["sqrt", "log2"])}),
    "Random Forest": ("Ensemble (Bagging)", RandomForestClassifier(n_jobs=1, random_state=RS),
        {"n_estimators": [200], "max_depth": [10, None], "min_samples_leaf": [1, 4]},
        {"n_estimators": ("int", 100, 300), "max_depth": ("int", 5, 40), "min_samples_leaf": ("int", 1, 10),
         "max_features": ("cat", ["sqrt", "log2"])}),
    "XGBoost": ("Ensemble (Gradient Boosting)",
        XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=RS),
        {"max_depth": [3, 6], "learning_rate": [0.05, 0.1], "n_estimators": [200]},
        {"max_depth": ("int", 2, 8), "learning_rate": ("log", 0.01, 0.3), "n_estimators": ("int", 100, 400),
         "subsample": ("float", 0.6, 1.0), "colsample_bytree": ("float", 0.5, 1.0), "reg_lambda": ("log", 0.1, 10)}),
    "CatBoost": ("Ensemble (Gradient Boosting)",
        CatBoostClassifier(iterations=300, verbose=0, thread_count=1, random_state=RS),
        {"depth": [4, 6], "learning_rate": [0.05, 0.1], "iterations": [200]},
        {"depth": ("int", 3, 8), "learning_rate": ("log", 0.01, 0.3), "iterations": ("int", 100, 400),
         "l2_leaf_reg": ("log", 1, 30)}),
    "AdaBoost": ("Ensemble (Boosting)",
        AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=RS),
        {"n_estimators": [100], "learning_rate": [0.1, 0.5, 1.0], "estimator__max_depth": [1, 2]},
        {"n_estimators": ("int", 50, 300), "learning_rate": ("log", 0.01, 1.5), "estimator__max_depth": ("int", 1, 3)}),
    "Gradient Boosting": ("Ensemble (Gradient Boosting)", GradientBoostingClassifier(random_state=RS),
        {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [2, 3], "subsample": [0.8]},
        {"n_estimators": ("int", 50, 200), "learning_rate": ("log", 0.01, 0.3), "max_depth": ("int", 1, 4),
         "subsample": ("float", 0.5, 1.0), "min_samples_leaf": ("int", 1, 20)}),
    "LightGBM": ("Ensemble (Gradient Boosting)", LGBMClassifier(verbose=-1, n_jobs=1, random_state=RS),
        {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1], "n_estimators": [200]},
        {"num_leaves": ("int", 7, 63), "learning_rate": ("log", 0.01, 0.3), "n_estimators": ("int", 100, 400),
         "subsample": ("float", 0.5, 1.0), "subsample_freq": ("cat", [1]), "colsample_bytree": ("float", 0.5, 1.0),
         "reg_lambda": ("log", 0.1, 10)}),
    "Voting": ("Ensemble (Voting: LR+LDA+SVM+ET)", VotingClassifier(ensemble_members(), voting="hard", n_jobs=1),
        {"lr__C": [0.1, 1], "svm__C": [0.01, 0.1]},
        {"lr__C": ("log", 1e-3, 10), "svm__C": ("log", 1e-3, 1), "et__min_samples_leaf": ("int", 1, 10)}),
    "Stacking": ("Ensemble (Stacking: LR+LDA+SVM+ET -> LR)",
        StackingClassifier(ensemble_members(), final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1),
        {"final_estimator__C": [0.1, 1]},
        {"final_estimator__C": ("log", 1e-3, 10), "lr__C": ("log", 1e-3, 10), "svm__C": ("log", 1e-3, 1),
         "et__min_samples_leaf": ("int", 1, 10)}),
    # LinearSVC: RBF SVC is intractable on >50k samples (O(n^2)–O(n^3)).
    "SVM": ("Kernel-Based (LinearSVC @ scale)", LinearSVC(dual="auto", random_state=RS, max_iter=5000),
        {"C": [0.1, 1, 10]},
        {"C": ("log", 0.01, 100)}),
    "Naive Bayes": ("Probabilistic", GaussianNB(),
        {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1, 10]},
        {"var_smoothing": ("log", 1e-10, 10)}),
    "Neural Network (MLP)": ("Neural Network",
        MLPClassifier(early_stopping=True, max_iter=200, random_state=RS),
        {"hidden_layer_sizes": [(128,), (256,)], "alpha": [1e-4, 1e-2]},
        {"hidden_layer_sizes": ("cat", [(64,), (128,), (256,), (128, 64)]), "alpha": ("log", 1e-5, 10),
         "learning_rate_init": ("log", 1e-4, 1e-2)}),
}


# ---------------------------------------------------------------- helpers
def pref(params):
    return {f"clf__{k}": v for k, v in params.items()}


def grid_for(grid, use_svd_k: bool):
    g = pref({k: v for k, v in grid.items()})
    if use_svd_k:
        g["svd__n_components"] = K_GRID
    return g


def scipy_space(space, use_svd_k: bool):
    d = {}
    for k, spec in space.items():
        t = spec[0]
        if t == "log":
            d[k] = loguniform(spec[1], spec[2])
        elif t == "float":
            d[k] = uniform(spec[1], spec[2] - spec[1])
        elif t == "int":
            d[k] = randint(spec[1], spec[2] + 1)
        else:
            d[k] = spec[1]
    d = pref(d)
    if use_svd_k:
        d["svd__n_components"] = randint(K_SPACE[1], K_SPACE[2] + 1)
    return d


def optuna_suggest(trial, space, use_svd_k: bool):
    p = {}
    for k, spec in space.items():
        t = spec[0]
        if t == "log":
            p[k] = trial.suggest_float(k, spec[1], spec[2], log=True)
        elif t == "float":
            p[k] = trial.suggest_float(k, spec[1], spec[2])
        elif t == "int":
            p[k] = trial.suggest_int(k, spec[1], spec[2])
        else:
            idx = trial.suggest_categorical(k, list(range(len(spec[1]))))
            p[k] = spec[1][idx]
    p = pref(p)
    if use_svd_k:
        p["svd__n_components"] = trial.suggest_int("svd__n_components", K_SPACE[1], K_SPACE[2])
    return p


def selection_score(train, val):
    gap = train - val
    if gap <= GAP_TARGET:
        return val - 0.01 * gap
    return val - gap - (1 - GAP_TARGET)


def fmt_val(v):
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


def fmt_params(p):
    clean = {k.replace("clf__", "").replace("svd__", "svd__"): v for k, v in p.items()}
    return ", ".join(f"{k}={fmt_val(v)}" for k, v in sorted(clean.items()))


def refit_rule(cv_results):
    tr = np.asarray(cv_results["mean_train_score"])
    va = np.asarray(cv_results["mean_test_score"])
    scores = [selection_score(a, b) for a, b in zip(tr, va)]
    return int(np.argmax(scores))


def log_rows(name, mode, params_list, tr, va, sd, chosen):
    rows = []
    for i, (p, a, b, s) in enumerate(zip(params_list, tr, va, sd)):
        gap = a - b
        rows.append({"Classifier": name, "Search": mode, "Params": fmt_params(p),
                     "Mean Train Acc": round(a, 4), "Mean CV Acc": round(b, 4), "Std CV Acc": round(s, 4),
                     "Train-Val Gap": round(gap, 4), "Within Gap Target": bool(gap <= GAP_TARGET),
                     "Selection Score": round(selection_score(a, b), 4), "Selected": i == chosen})
    return rows


def scores_for_auc(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    # hard-voting ensembles etc.
    return model.predict(X).astype(float)


def risk_and_status(train_acc, tt_gap, skf_gap):
    g = max(tt_gap, skf_gap)
    risk = "HIGH" if g > 0.2 else "MODERATE" if g > 0.1 else "LOW"
    if risk == "HIGH":
        status = "OVERFITTING"
    elif train_acc < 0.5:
        status = "UNDERFITTING"
    elif risk == "MODERATE":
        status = "GOOD FIT (moderate gap)"
    else:
        status = "GOOD FIT"
    return risk, status


# ---------------------------------------------------------------- tuning
def tune(name, mode, pipe, grid, space, Xtr, ytr, inner, n_iter, use_svd_k: bool):
    logs = []
    if mode == "Default":
        return pipe, "default", "N/A", "N/A", logs
    if mode in ("Grid Search", "Random Search"):
        if mode == "Grid Search":
            search = GridSearchCV(pipe, grid_for(grid, use_svd_k), cv=inner, scoring="accuracy",
                                  return_train_score=True, refit=refit_rule, n_jobs=N_JOBS)
        else:
            search = RandomizedSearchCV(pipe, scipy_space(space, use_svd_k), n_iter=n_iter, cv=inner,
                                        scoring="accuracy", return_train_score=True, refit=refit_rule,
                                        random_state=RS, n_jobs=N_JOBS)
        search.fit(Xtr, ytr)
        r = search.cv_results_
        i = search.best_index_
        logs = log_rows(name, mode, r["params"], r["mean_train_score"], r["mean_test_score"],
                        r["std_test_score"], i)
        return (search.best_estimator_, fmt_params(r["params"][i]),
                round(r["mean_train_score"][i], 4), round(r["mean_test_score"][i], 4), logs)

    trials = []

    def objective(trial):
        p = optuna_suggest(trial, space, use_svd_k)
        m = clone(pipe).set_params(**p)
        cv = cross_validate(m, Xtr, ytr, cv=inner, scoring="accuracy",
                            return_train_score=True, n_jobs=N_JOBS)
        a, b, s = cv["train_score"].mean(), cv["test_score"].mean(), cv["test_score"].std()
        trials.append((p, a, b, s))
        return selection_score(a, b)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(objective, n_trials=n_iter)
    best = int(np.argmax([selection_score(t[1], t[2]) for t in trials]))
    p, a, b, _ = trials[best]
    logs = log_rows(name, mode, [t[0] for t in trials], [t[1] for t in trials],
                    [t[2] for t in trials], [t[3] for t in trials], best)
    model = clone(pipe).set_params(**p).fit(Xtr, ytr)
    return model, fmt_params(p), round(a, 4), round(b, 4), logs


# ---------------------------------------------------------------- evaluation
def evaluate(model, Xtr, ytr, Xte, yte, kf, skf, skip_cv: bool = False):
    train_acc = accuracy_score(ytr, model.predict(Xtr))
    pred = model.predict(Xte)
    test_acc = accuracy_score(yte, pred)
    try:
        auc = roc_auc_score(yte, scores_for_auc(model, Xte))
    except Exception:
        auc = np.nan
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    if skip_cv:
        # Memory-safe path on huge matrices: report hold-out metrics only.
        return {
            "train_acc": train_acc, "test_acc": test_acc, "pred": pred,
            "prec": precision_score(yte, pred, average="macro", zero_division=0),
            "rec": recall_score(yte, pred, average="macro", zero_division=0),
            "f1": f1_score(yte, pred, average="macro", zero_division=0),
            "auc": auc, "cm": cm,
            "kf_mean": test_acc, "kf_std": 0.0,
            "skf_mean": test_acc, "skf_std": 0.0,
            "skf_train": train_acc,
        }
    k = cross_validate(clone(model), Xtr, ytr, cv=kf, scoring="accuracy", n_jobs=N_JOBS)
    s = cross_validate(clone(model), Xtr, ytr, cv=skf, scoring="accuracy",
                       return_train_score=True, n_jobs=N_JOBS)
    return {
        "train_acc": train_acc, "test_acc": test_acc, "pred": pred,
        "prec": precision_score(yte, pred, average="macro", zero_division=0),
        "rec": recall_score(yte, pred, average="macro", zero_division=0),
        "f1": f1_score(yte, pred, average="macro", zero_division=0),
        "auc": auc, "cm": cm,
        "kf_mean": k["test_score"].mean(), "kf_std": k["test_score"].std(),
        "skf_mean": s["test_score"].mean(), "skf_std": s["test_score"].std(),
        "skf_train": s["train_score"].mean(),
    }


def result_row(name, cat, mode, params, in_tr, in_va, ev, secs, n_test, feature_name):
    tt_gap = ev["train_acc"] - ev["test_acc"]
    skf_gap = ev["skf_train"] - ev["skf_mean"]
    risk, status = risk_and_status(ev["train_acc"], tt_gap, skf_gap)
    r4 = lambda x: round(float(x), 4)
    return {
        "Feature Set": feature_name,
        "Classifier Name": name, "Category": cat, "Tuning": mode, "Best Hyperparameters": params,
        "Inner CV Train Acc": in_tr, "Inner CV Val Acc (Search)": in_va,
        "Training Accuracy *": r4(ev["train_acc"]), "Training Error": r4(1 - ev["train_acc"]),
        "Test Accuracy": r4(ev["test_acc"]), "Test Error": r4(1 - ev["test_acc"]),
        "Precision (Macro)": r4(ev["prec"]), "Recall (Macro)": r4(ev["rec"]),
        "F1-Score (Macro)": r4(ev["f1"]), "ROC-AUC (Macro OvR)": r4(ev["auc"]),
        "Confusion Matrix (Correct/Total)": f"{int(np.trace(ev['cm']))}/{n_test}",
        "K-Fold CV (5)": f"{ev['kf_mean']:.4f} +/- {ev['kf_std']:.4f}", "K-Fold Mean": r4(ev["kf_mean"]),
        "Stratified K-Fold CV (5) *": f"{ev['skf_mean']:.4f} +/- {ev['skf_std']:.4f}",
        "Stratified K-Fold Mean": r4(ev["skf_mean"]), "Stratified K-Fold Std": r4(ev["skf_std"]),
        "SKF Train Acc": r4(ev["skf_train"]), "Train-Test Gap": r4(tt_gap), "SKF Train-Val Gap": r4(skf_gap),
        "Overfitting Risk": risk, "Fit Status": status, "Fit/Search Time (sec)": round(secs, 2),
    }


def rank_df(df):
    df = df.copy()
    df["_high"] = (df["Overfitting Risk"] == "HIGH").astype(int)
    df = df.sort_values(["_high", "Stratified K-Fold Mean", "Test Accuracy"], ascending=[True, False, False])
    return df.drop(columns="_high")


# ---------------------------------------------------------------- excel
def write_excel(path, all_df, logs_df, cms, info, always_df, vote_df, ge5_df=None):
    best = rank_df(all_df).groupby("Classifier Name", sort=False).head(1)
    best = rank_df(best).reset_index(drop=True)
    best.insert(0, "Rank", range(1, len(best) + 1))
    over_cols = ["Rank", "Classifier Name", "Tuning", "Training Accuracy *", "Test Accuracy", "Train-Test Gap",
                 "SKF Train Acc", "Stratified K-Fold Mean", "SKF Train-Val Gap", "Overfitting Risk", "Fit Status"]
    modes = ["Default", "Grid Search", "Random Search", "Bayesian Optimization"]
    order = list(dict.fromkeys(all_df["Classifier Name"]))

    def pivot(col):
        p = all_df.pivot(index="Classifier Name", columns="Tuning", values=col)
        p = p.reindex(index=order, columns=[m for m in modes if m in p.columns])
        return p.reset_index()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        best.to_excel(xw, sheet_name="Best per Classifier", index=False)
        best[over_cols].to_excel(xw, sheet_name="Overfitting Check", index=False)
        all_df.to_excel(xw, sheet_name="All Results", index=False)
        for sheet, col in [("Tuning - Test Accuracy", "Test Accuracy"), ("Tuning - F1-Score", "F1-Score (Macro)"),
                           ("Tuning - Stratified K-Fold", "Stratified K-Fold Mean"),
                           ("Tuning - Training Accuracy", "Training Accuracy *"),
                           ("Tuning - Train-Test Gap", "Train-Test Gap"), ("Tuning - Fit Status", "Fit Status")]:
            pivot(col).to_excel(xw, sheet_name=sheet, index=False)
        if len(logs_df):
            logs_df.to_excel(xw, sheet_name="Search Logs", index=False)
        for name in best["Classifier Name"]:
            if name not in cms:
                continue
            cm = cms[name]
            pd.DataFrame(cm, index=[f"T:{l}" for l in LABELS], columns=[f"P:{l}" for l in LABELS]) \
                .to_excel(xw, sheet_name=f"CM - {name}"[:31])
        if always_df is not None and len(always_df):
            always_df.to_excel(xw, sheet_name="Always Misclassified", index=False)
        if ge5_df is not None and len(ge5_df):
            ge5_df.to_excel(xw, sheet_name="Misclass by >=5 Clfs", index=False)
        elif ge5_df is not None:
            pd.DataFrame({"note": ["No claim was misclassified by >=5 classifiers."]}).to_excel(
                xw, sheet_name="Misclass by >=5 Clfs", index=False
            )
        if vote_df is not None and len(vote_df):
            # full vote table can be huge; keep lean columns for Excel sheet
            lean_cols = [c for c in vote_df.columns if not str(c).startswith("wrong:")]
            vote_df[lean_cols].to_excel(xw, sheet_name="Misclass Vote Counts", index=False)
        pd.DataFrame(list(info.items()), columns=["Property", "Value"]).to_excel(xw, sheet_name="Dataset Info", index=False)
        pd.DataFrame({"Encoded": [0, 1], "Original": LABELS}).to_excel(xw, sheet_name="Class Labels", index=False)

        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
            ws.freeze_panes = "A2"


def write_master_misclassified(path, frames, summary_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name="Summary", index=False)
        for feat, df in frames.items():
            sheet = f"{feat}"[:31]
            if df is None or not len(df):
                pd.DataFrame({"note": ["No claim was misclassified by ALL classifiers."]}).to_excel(
                    xw, sheet_name=sheet, index=False)
            else:
                df.to_excel(xw, sheet_name=sheet, index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 80)
            ws.freeze_panes = "A2"


# ---------------------------------------------------------------- data + embeddings
def load_balanced_frame(data_path: str, sample: int, balance: bool):
    df = pd.read_csv(data_path)
    n_raw = len(df)
    df = df[["id", "claim", "verifiable"]].drop_duplicates(subset="claim")
    n_dedup = len(df)
    if balance:
        per_class = int(df["verifiable"].value_counts().min())
        if sample:
            per_class = min(per_class, sample // 2)
        df = df.groupby("verifiable", group_keys=False).sample(n=per_class, random_state=RS)
    elif sample and sample < len(df):
        df, _ = train_test_split(df, train_size=sample, stratify=df["verifiable"], random_state=RS)
    df = df.reset_index(drop=True)
    df["label"] = (df["verifiable"] == "non-factual").astype(np.int64)
    return df, n_raw, n_dedup


def _load_hf_or_local(preferred: str, local_fallbacks: list[str]) -> str:
    """Prefer cached HuggingFace id; fall back to local DistilBERT when downloads fail."""
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(preferred, local_files_only=True)
        return preferred
    except Exception:
        pass
    try:
        AutoConfig.from_pretrained(preferred, local_files_only=False)
        return preferred
    except Exception as exc:
        print(f"Could not fetch '{preferred}' ({exc.__class__.__name__}). Trying local fallbacks...")
    for name in local_fallbacks:
        try:
            AutoConfig.from_pretrained(name, local_files_only=True)
            print(f"NOTE: using local '{name}' instead of '{preferred}'.")
            return name
        except Exception:
            continue
    raise RuntimeError(f"Could not load {preferred} or fallbacks {local_fallbacks}")


def _encode_sentence_transformer(model_name: str, texts: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except TypeError:
        model = SentenceTransformer(model_name, device=device)
    except Exception:
        model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def _encode_mean_pool_transformer(model_name: str, texts: list[str], batch_size: int = 16,
                                  pool: str = "mean") -> np.ndarray:
    """Mean-pool or CLS last hidden states for BERT / RoBERTa / DistilBERT."""
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
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            h = model(**enc).last_hidden_state
            if pool == "cls":
                pooled = h[:, 0, :]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).expand(h.size()).float()
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.cpu().numpy().astype(np.float32))
            if (i // batch_size) % 50 == 0:
                print(f"  encoded {min(i + batch_size, len(texts)):,}/{len(texts):,}", flush=True)
    del model
    return np.vstack(outs)


def try_reuse_minilm(df: pd.DataFrame, reuse_dir: Path):
    """Map balanced rows to precomputed MiniLM vectors by claim id."""
    emb_path = reuse_dir / "claim_embeddings.npy"
    meta_path = reuse_dir / "metadata.csv"
    if not emb_path.exists() or not meta_path.exists():
        return None
    meta = pd.read_csv(meta_path, usecols=["id"])
    if len(meta) == 0:
        return None
    id_to_row = pd.Series(np.arange(len(meta)), index=meta["id"].astype(np.int64))
    missing = (~df["id"].astype(np.int64).isin(id_to_row.index)).sum()
    if missing:
        print(f"MiniLM reuse: {missing} ids missing; will re-encode.")
        return None
    X_all = np.load(emb_path, mmap_mode="r")
    idx = id_to_row.loc[df["id"].astype(np.int64)].to_numpy()
    print(f"Reusing MiniLM cache ({reuse_dir}) for {len(idx):,} rows.")
    return np.asarray(X_all[idx], dtype=np.float32)


def build_tfidf_features(train_texts, test_texts):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    svd = TruncatedSVD(n_components=MAX_SVD, random_state=RS)
    Xtr = svd.fit_transform(tfidf.fit_transform(train_texts)).astype(np.float32)
    Xte = svd.transform(tfidf.transform(test_texts)).astype(np.float32)
    var = float(svd.explained_variance_ratio_.sum())
    info = {
        "Vectoriser": "TF-IDF (1-2 grams, max 50k, sublinear_tf)",
        "Reduction": f"TruncatedSVD {MAX_SVD} components",
        "Explained variance": f"{var:.1%}",
        "Vocabulary size": len(tfidf.vocabulary_),
        "use_svd_k": True,
        "dim": MAX_SVD,
    }
    return Xtr, Xte, info


def build_dense_features(key: str, train_texts, test_texts, train_df, test_df):
    cfg = EMBEDDERS[key]
    if key == "minilm" and cfg.get("reuse_project_cache"):
        # Prefer encoding the split texts for correctness of train-fit leakage rules
        # on TF-IDF; for frozen ST models, reuse by id on the concatenated frame.
        full = pd.concat([train_df, test_df], axis=0)
        reused = try_reuse_minilm(full, cfg["reuse_project_cache"])
        if reused is not None:
            ntr = len(train_df)
            Xtr, Xte = reused[:ntr], reused[ntr:]
            info = {
                "Vectoriser": cfg["label"],
                "Reduction": "none (frozen sentence embedding)",
                "Explained variance": "N/A",
                "Embedding dim": reused.shape[1],
                "Normalized": True,
                "Source": f"reused {cfg['reuse_project_cache']}",
                "use_svd_k": False,
                "dim": reused.shape[1],
            }
            return Xtr, Xte, info

    texts = list(train_texts) + list(test_texts)
    local_bertish = "typeform/distilbert-base-uncased-mnli"
    used_model = cfg["model"]
    used_label = cfg["label"]
    pool = "mean"

    print(f"Encoding {len(texts):,} claims with {cfg['label']} ...", flush=True)
    if cfg["kind"] == "sentence_transformer":
        try:
            emb = _encode_sentence_transformer(cfg["model"], texts)
        except Exception as exc:
            print(f"NOTE: {cfg['model']} unavailable ({exc.__class__.__name__}); "
                  f"using DistilBERT mean-pool fallback.")
            used_model = _load_hf_or_local(cfg["model"], [local_bertish])
            used_label = f"DistilBERT mean-pool ({key} slot fallback)"
            emb = _encode_mean_pool_transformer(used_model, texts, pool="mean")
    else:
        if key == "bert":
            used_model = _load_hf_or_local(cfg["model"], [local_bertish])
            if not used_model.startswith("bert"):
                used_label = "DistilBERT mean-pool (BERT-family fallback)"
        elif key == "roberta":
            used_model = _load_hf_or_local(cfg["model"], [local_bertish])
            if "roberta" not in used_model:
                used_label = "DistilBERT CLS-pool (RoBERTa slot; HF blocked)"
                pool = "cls"
        emb = _encode_mean_pool_transformer(used_model, texts, pool=pool)

    ntr = len(train_texts)
    Xtr, Xte = emb[:ntr], emb[ntr:]
    info = {
        "Vectoriser": used_label,
        "Model id": used_model,
        "Reduction": "none (frozen transformer embedding, mean-pool / ST)",
        "Explained variance": "N/A",
        "Embedding dim": emb.shape[1],
        "Normalized": True,
        "use_svd_k": False,
        "dim": emb.shape[1],
    }
    return Xtr, Xte, info


def feature_cache_dir(cache_dir: Path, feature_key: str, n_rows: int) -> Path:
    return Path(cache_dir) / f"{feature_key}_n{n_rows}_rs{RS}"


def save_feature_cache(folder: Path, Xtr, Xte, ytr, yte, test_meta: pd.DataFrame, feat_info: dict):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "Xtr.npy", Xtr)
    np.save(folder / "Xte.npy", Xte)
    np.save(folder / "ytr.npy", ytr)
    np.save(folder / "yte.npy", yte)
    test_meta.to_csv(folder / "test_meta.csv", index=False)
    serializable = {k: v for k, v in feat_info.items() if k != "use_svd_k"}
    serializable["use_svd_k"] = bool(feat_info["use_svd_k"])
    (folder / "feat_info.json").write_text(json.dumps(serializable, indent=2))


def load_feature_cache(folder: Path):
    Xtr = np.load(folder / "Xtr.npy", mmap_mode="r")
    Xte = np.load(folder / "Xte.npy", mmap_mode="r")
    ytr = np.load(folder / "ytr.npy")
    yte = np.load(folder / "yte.npy")
    test_meta = pd.read_csv(folder / "test_meta.csv")
    feat_info = json.loads((folder / "feat_info.json").read_text())
    return Xtr, Xte, ytr, yte, test_meta, feat_info


def prepare_features(feature_key: str, df: pd.DataFrame, cache_dir: Path, force: bool = False):
    folder = feature_cache_dir(cache_dir, feature_key, len(df))
    marker = folder / "Xtr.npy"
    if marker.exists() and not force:
        print(f"Loading cached features: {folder}")
        return load_feature_cache(folder)

    y = df["label"].to_numpy()
    idx = np.arange(len(df))
    i_tr, i_te, ytr, yte = train_test_split(idx, y, test_size=0.2, stratify=y, random_state=RS)
    train_df = df.iloc[i_tr].reset_index(drop=True)
    test_df = df.iloc[i_te].reset_index(drop=True)
    t_tr = train_df["claim"].astype(str).tolist()
    t_te = test_df["claim"].astype(str).tolist()

    if feature_key == "tfidf":
        Xtr, Xte, feat_info = build_tfidf_features(t_tr, t_te)
        feature_name = f"TF-IDF + SVD-{MAX_SVD}"
    else:
        Xtr, Xte, feat_info = build_dense_features(feature_key, t_tr, t_te, train_df, test_df)
        feature_name = feat_info.get("Vectoriser", EMBEDDERS[feature_key]["label"])

    feat_info["feature_name"] = feature_name
    test_meta = test_df[["id", "claim", "verifiable", "label"]].copy()
    save_feature_cache(folder, Xtr, Xte, ytr, yte, test_meta, feat_info)
    print(f"Train {Xtr.shape}, Test {Xte.shape}  [{feature_name}]")
    return Xtr, Xte, ytr, yte, test_meta, feat_info


def preds_dir(cache_dir: Path, feature_key: str) -> Path:
    return Path(cache_dir) / f"preds_{feature_key}"


def save_best_pred(cache_dir: Path, feature_key: str, clf_name: str, pred: np.ndarray, meta: dict):
    d = preds_dir(cache_dir, feature_key)
    d.mkdir(parents=True, exist_ok=True)
    safe = clf_name.replace("/", "_").replace(" ", "_")
    np.save(d / f"{safe}.npy", pred.astype(np.int8))
    (d / f"{safe}.json").write_text(json.dumps(meta, indent=2))


def load_all_best_preds(cache_dir: Path, feature_key: str) -> dict:
    d = preds_dir(cache_dir, feature_key)
    out = {}
    if not d.exists():
        return out
    for p in d.glob("*.npy"):
        meta_path = p.with_suffix(".json")
        name = p.stem.replace("_", " ")
        if meta_path.exists():
            try:
                name = json.loads(meta_path.read_text()).get("classifier", name)
            except Exception:
                pass
        out[name] = np.load(p)
    return out


def _label_name(y):
    return np.where(np.asarray(y) == 0, "factual", "non-factual")


def misclassification_tables(test_meta: pd.DataFrame, yte, best_preds: dict, min_wrong: int = 5):
    """best_preds: classifier_name -> y_pred on the shared test set.

    Returns:
      always_df  — wrong for EVERY classifier
      vote_df    — full per-classifier vote table
      ge5_df     — wrong for at least `min_wrong` classifiers (true vs majority predicted)
    """
    names = list(best_preds.keys())
    if not names:
        empty = pd.DataFrame()
        return empty, empty, empty
    P = np.vstack([best_preds[n] for n in names])  # C x N
    wrong = P != yte[None, :]
    n_wrong = wrong.sum(axis=0)
    always = n_wrong == len(names)
    # majority vote prediction (ties broken toward non-factual=1)
    maj = (P.sum(axis=0) >= (len(names) / 2.0)).astype(np.int64)

    vote = test_meta.copy()
    vote["true_label"] = _label_name(yte)
    vote["predicted_label_majority"] = _label_name(maj)
    vote["n_classifiers_wrong"] = n_wrong
    vote["n_classifiers_total"] = len(names)
    vote["frac_classifiers_wrong"] = (n_wrong / len(names)).round(4)
    wrong_name_lists = []
    wrong_pred_labels = []
    for j in range(P.shape[1]):
        bad_idx = [i for i in range(len(names)) if wrong[i, j]]
        wrong_name_lists.append(", ".join(names[i] for i in bad_idx))
        if bad_idx:
            # mode of wrong predictions (binary => opposite of true)
            vals = P[bad_idx, j]
            wrong_pred_labels.append("non-factual" if vals.sum() >= (len(vals) / 2.0) else "factual")
        else:
            wrong_pred_labels.append("")
    vote["classifiers_that_misclassified"] = wrong_name_lists
    vote["predicted_label"] = wrong_pred_labels  # what the misclassifying models predicted
    for i, n in enumerate(names):
        vote[f"pred:{n}"] = _label_name(best_preds[n])
        vote[f"wrong:{n}"] = wrong[i]

    # Filter on column values — never positional boolean on a sorted frame
    always_df = vote.loc[vote["n_classifiers_wrong"] == len(names),
                         ["id", "claim", "true_label", "predicted_label",
                          "n_classifiers_wrong"]].copy()
    always_df.insert(0, "n_classifiers", len(names))
    always_df = always_df.reset_index(drop=True)

    ge5_df = vote.loc[vote["n_classifiers_wrong"] >= int(min_wrong),
                      ["id", "claim", "true_label", "predicted_label",
                       "predicted_label_majority", "n_classifiers_wrong",
                       "n_classifiers_total", "classifiers_that_misclassified"]].copy()
    ge5_df.insert(0, "min_wrong_threshold", int(min_wrong))
    ge5_df = ge5_df.sort_values(["n_classifiers_wrong", "id"], ascending=[False, True]).reset_index(drop=True)

    vote = vote.sort_values(["n_classifiers_wrong", "id"], ascending=[False, True]).reset_index(drop=True)
    return always_df, vote, ge5_df


# ---------------------------------------------------------------- per-feature experiment
def _results_csv(out_path: Path) -> Path:
    return out_path.with_suffix(".csv")


def _load_existing_results(out_path: Path):
    csv_path = _results_csv(out_path)
    # Prefer CSV sidecar (atomic-ish); Excel can be corrupt if the process is killed mid-write.
    for path, reader in (
        (csv_path, lambda p: pd.read_csv(p)),
        (out_path, lambda p: pd.read_excel(p, sheet_name="All Results")),
    ):
        if not path.exists():
            continue
        try:
            all_df = reader(path)
            rows = all_df.to_dict(orient="records")
            logs = []
            if out_path.exists():
                try:
                    logs = pd.read_excel(out_path, sheet_name="Search Logs").to_dict(orient="records")
                except Exception:
                    pass
            print(f"Loaded {len(rows)} existing rows from {path.name}")
            return rows, logs, {}, {}
        except Exception as exc:
            print(f"Could not resume from {path}: {exc}")
    return [], [], {}, {}


def run_feature(feature_key, df, n_raw, n_dedup, args, modes, names, out_dir: Path, cache_dir: Path):
    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    Xtr, Xte, ytr, yte, test_meta, feat_info = prepare_features(
        feature_key, df, cache_dir, force=args.recompute_features
    )
    use_svd_k = bool(feat_info.get("use_svd_k", feature_key == "tfidf"))
    feature_name = feat_info.get("feature_name", feature_key)
    skip_cv = bool(args.skip_cv)

    inner = StratifiedKFold(args.inner, shuffle=True, random_state=RS)
    kf = KFold(5, shuffle=True, random_state=RS)
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)

    out_path = out_dir / f"results_{feature_key}.xlsx"
    rows, logs, cms, best_preds = [], [], {}, {}
    best_seen = {}
    done = set()

    if args.resume:
        rows, logs, cms, best_preds = _load_existing_results(out_path)
        for r in rows:
            done.add((r.get("Classifier Name"), r.get("Tuning")))
        if done:
            print(f"Resuming {feature_key}: {len(done)} classifier/mode combos already done.")

    for name in names:
        cat, clf, grid, space = CLASSIFIERS[name]
        for mode in modes:
            if (name, mode) in done:
                continue
            t0 = time.time()
            pipe = make_pipe(clf, use_svd_k=use_svd_k)
            model, params, in_tr, in_va, lg = tune(
                name, mode, pipe, grid, space, Xtr, ytr, inner, args.n_iter, use_svd_k
            )
            if mode == "Default":
                model = pipe.fit(Xtr, ytr)
            secs = time.time() - t0
            ev = evaluate(model, Xtr, ytr, Xte, yte, kf, skf, skip_cv=skip_cv)
            row = result_row(name, cat, mode, params, in_tr, in_va, ev, secs, len(yte), feature_name)
            rows.append(row)
            logs.extend(lg)
            key = (row["Overfitting Risk"] == "HIGH", -row["Stratified K-Fold Mean"], -row["Test Accuracy"])
            if name not in best_seen or key < best_seen[name]:
                best_seen[name] = key
                cms[name] = ev["cm"]
                best_preds[name] = ev["pred"]
                save_best_pred(cache_dir, feature_key, name, ev["pred"], {
                    "classifier": name, "tuning": mode,
                    "test_accuracy": row["Test Accuracy"],
                    "skf_mean": row["Stratified K-Fold Mean"],
                    "overfitting_risk": row["Overfitting Risk"],
                })
            print(f"[{feature_key}] {name:22s} {mode:22s} test={row['Test Accuracy']:.4f} "
                  f"f1={row['F1-Score (Macro)']:.4f} skf={row['Stratified K-Fold Mean']:.4f} "
                  f"{row['Fit Status']} ({secs:.0f}s)", flush=True)

            disk_preds = load_all_best_preds(cache_dir, feature_key)
            disk_preds.update(best_preds)
            all_df = pd.DataFrame(rows)
            always_df, vote_df, ge5_df = misclassification_tables(test_meta, yte, disk_preds)
            info = {
                "Data File": Path(args.data).name,
                "Dataset": "FEVER (claim text only)",
                "Feature Set": feature_name,
                "Raw Rows": n_raw,
                "Rows After Dedup": n_dedup,
                "Rows Used": f"{len(df)} (balanced 50/50 sample)",
                "Features": "; ".join(f"{k}={v}" for k, v in feat_info.items()
                                      if k not in ("use_svd_k", "feature_name")),
                "Num Classes": 2,
                "Class Balance": f"factual {np.mean(ytr == 0):.1%} train / non-factual {np.mean(ytr == 1):.1%} train",
                "Majority-Class Baseline Acc": round(float(np.mean(yte == 0)), 4),
                "Train Size": len(ytr), "Test Size": len(yte), "Split Ratio": "80:20", "Stratified Split": "Yes",
                "K-Fold CV": ("skipped (hold-out proxy)" if skip_cv else "5-Fold (shuffled, on training set)"),
                "Stratified K-Fold CV": ("skipped (hold-out proxy)" if skip_cv else "5-Fold (shuffled, on training set)"),
                "Tuning Inner CV": f"Stratified {args.inner}-Fold on training set",
                "Random Search Iterations": args.n_iter,
                "Bayesian Optimization": f"Optuna TPE, {args.n_iter} trials",
                "Tuning Objective": f"Best val accuracy among settings with inner train-val gap <= {GAP_TARGET}",
                "Always Misclassified": f"{len(always_df)} test claims wrong by all {len(disk_preds)} classifiers",
                "Misclassified by >=5": f"{len(ge5_df)} test claims wrong by >=5 classifiers",
                "SVM Note": "LinearSVC used instead of RBF SVC for tractability on large samples",
                "Random State": RS,
            }
            all_df.to_csv(_results_csv(out_path), index=False)
            always_df.to_csv(out_dir / f"always_misclassified_{feature_key}.csv", index=False)
            ge5_df.to_csv(out_dir / f"misclassified_ge5_{feature_key}.csv", index=False)
            # Excel only after each classifier finishes all modes (avoids lock/corruption)
            modes_for_clf = [m for m in modes]
            if mode == modes_for_clf[-1] or all((name, m) in done or m == mode for m in modes_for_clf):
                try:
                    write_excel(out_path, all_df, pd.DataFrame(logs) if logs else pd.DataFrame(),
                                cms, info, always_df, vote_df, ge5_df)
                except Exception as exc:
                    print(f"Excel write failed (CSV kept): {exc}", flush=True)
            done.add((name, mode))
            del model, pipe, ev
            gc.collect()

    disk_preds = load_all_best_preds(cache_dir, feature_key)
    disk_preds.update(best_preds)
    always_df, vote_df, ge5_df = misclassification_tables(test_meta, yte, disk_preds)
    if len(rows):
        all_df = pd.DataFrame(rows)
        write_excel(
            out_path, all_df, pd.DataFrame(logs) if logs else pd.DataFrame(), cms,
            {
                "Data File": Path(args.data).name,
                "Feature Set": feature_name,
                "Rows Used": len(df),
                "Always Misclassified": f"{len(always_df)} / test {len(yte)} "
                                        f"(from {len(disk_preds)} classifiers)",
                "Misclassified by >=5": f"{len(ge5_df)} / test {len(yte)}",
                "Train Size": len(ytr), "Test Size": len(yte),
                "Classifiers with preds": ", ".join(sorted(disk_preds)),
            },
            always_df, vote_df, ge5_df,
        )
        always_df.to_csv(out_dir / f"always_misclassified_{feature_key}.csv", index=False)
        ge5_df.to_csv(out_dir / f"misclassified_ge5_{feature_key}.csv", index=False)

    print(f"Saved {out_path} | always-misclassified: {len(always_df)} | "
          f"ge5-misclassified: {len(ge5_df)} (preds from {len(disk_preds)} classifiers)")
    return {
        "feature_key": feature_key,
        "feature_name": feature_name,
        "out_path": str(out_path),
        "all_df": pd.DataFrame(rows),
        "always_df": always_df,
        "ge5_df": ge5_df,
        "n_always": len(always_df),
        "n_ge5": len(ge5_df),
        "n_test": len(yte),
        "n_classifiers": max(len(disk_preds),
                             pd.DataFrame(rows)["Classifier Name"].nunique() if rows else 0),
    }


def run_experiment(
    feature_key: str,
    sample: int,
    out_dir: str | Path,
    cache_dir: str | Path | None = None,
    data: str = DEFAULT_DATA,
    modes: str = "Default,Grid Search,Random Search,Bayesian Optimization",
    n_iter: int = 10,
    inner: int = 3,
    only: str = "",
    resume: bool = True,
    skip_cv: bool | None = None,
    recompute_features: bool = False,
):
    """
    Run all (or selected) classifiers for one embedder / feature set.

    sample:
      4000  -> 2k factual + 2k non-factual
      0     -> full minority class (~67,134 balanced)
    """
    if feature_key != "tfidf" and feature_key not in EMBEDDERS:
        raise ValueError(f"Unknown feature '{feature_key}'. Use tfidf or {list(EMBEDDERS)}")

    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir) if cache_dir else (out_dir / "cache")
    if skip_cv is None:
        skip_cv = sample == 0 or sample >= 20000

    args = SimpleNamespace(
        data=data,
        sample=sample,
        balance=True,
        n_iter=n_iter,
        inner=inner,
        resume=resume,
        skip_cv=skip_cv,
        recompute_features=recompute_features,
    )
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    names = [n.strip() for n in only.split(",")] if only else list(CLASSIFIERS)

    df, n_raw, n_dedup = load_balanced_frame(data, sample, True)
    print(f"Balanced rows: {len(df):,} "
          f"(factual={(df.verifiable == 'factual').sum():,}, "
          f"non-factual={(df.verifiable == 'non-factual').sum():,})")
    print(f"Feature: {feature_key} | out: {out_dir}", flush=True)

    result = run_feature(feature_key, df, n_raw, n_dedup, args, mode_list, names, out_dir, cache_dir)
    print(f"\nDone. Excel: {result['out_path']}")
    return result
