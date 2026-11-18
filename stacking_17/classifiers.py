"""
Classifier suite matching fever_easy (17 names).

Stacking bases = 15 individual learners (exclude Voting & nested Stacking).
Meta candidates = clones of those same individuals (top-K by validation rank).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

RS = 42

# Names that appear in the 17-classifier faculty suite but are NOT used as bases
# of a new stacker (would nest ensembles).
EXCLUDE_AS_BASE = {"Voting", "Stacking"}


def _base_estimators(
    random_state: int = RS, *, large_data: bool = False
) -> dict[str, BaseEstimator]:
    """All individual classifiers from the 17-suite (no Voting/Stacking)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    est: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Perceptron": Perceptron(random_state=random_state),
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.5),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=random_state),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=60 if large_data else 80,
            n_jobs=1,
            random_state=random_state,
            max_depth=16 if large_data else 20,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=60 if large_data else 80,
            n_jobs=1,
            random_state=random_state,
            max_depth=16 if large_data else 20,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=30 if large_data else 40,
            random_state=random_state,
        ),
        # HistGradientBoosting is much faster than classic GB on large n.
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=4, max_iter=80 if large_data else 100, random_state=random_state
        ),
        "SVM": LinearSVC(dual="auto", random_state=random_state, max_iter=3000),
        "Naive Bayes": GaussianNB(),
    }
    # KNN / MLP are O(n²)/heavy on ~50k rows — keep on 4k, skip on 67k.
    if not large_data:
        est["KNN"] = KNeighborsClassifier(
            n_neighbors=11, algorithm="ball_tree", leaf_size=40, n_jobs=1
        )
        est["Neural Network (MLP)"] = MLPClassifier(
            hidden_layer_sizes=(64,),
            early_stopping=True,
            max_iter=120,
            random_state=random_state,
        )
    try:
        from xgboost import XGBClassifier

        est["XGBoost"] = XGBClassifier(
            tree_method="hist",
            eval_metric="logloss",
            n_estimators=100,
            max_depth=3,
            n_jobs=1,
            random_state=random_state,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        est["LightGBM"] = LGBMClassifier(
            verbose=-1, n_estimators=100, n_jobs=1, random_state=random_state
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier

        est["CatBoost"] = CatBoostClassifier(
            iterations=100, verbose=0, thread_count=1, random_state=random_state
        )
    except Exception:
        pass
    return est


def suite_names(random_state: int = RS) -> list[str]:
    """Full 17-suite name list for reporting (includes Voting/Stacking labels)."""
    bases = list(_base_estimators(random_state, large_data=False).keys())
    return bases + ["Voting", "Stacking"]


def get_base_pool(
    random_state: int = RS, *, large_data: bool = False
) -> dict[str, BaseEstimator]:
    """Estimators used as stacking level-0 bases."""
    return _base_estimators(random_state, large_data=large_data)


def wrap_for_proba(name: str, est: BaseEstimator) -> BaseEstimator:
    """
    Return a clone suitable as a stacking base / meta.

    Do NOT wrap with CalibratedClassifierCV here — that nests CV inside
    StackingClassifier CV and is extremely slow. sklearn StackingClassifier
    already accepts decision_function when predict_proba is missing.
    """
    return clone(est)


def make_meta_estimator(
    name: str, random_state: int = RS, *, large_data: bool = False
) -> BaseEstimator:
    """Clone a named classifier for use as stacking meta (final) model."""
    pool = _base_estimators(random_state, large_data=large_data)
    if name not in pool:
        raise KeyError(f"Meta model '{name}' not in base pool: {list(pool)}")
    return wrap_for_proba(name, pool[name])


def stacking_estimators_list(
    random_state: int = RS, *, large_data: bool = False
) -> list[tuple[str, BaseEstimator]]:
    """List of (short_name, estimator) for sklearn StackingClassifier."""
    out = []
    for i, (name, est) in enumerate(get_base_pool(random_state, large_data=large_data).items()):
        key = f"b{i}_{name.replace(' ', '_').replace('(', '').replace(')', '')[:20]}"
        out.append((key, wrap_for_proba(name, est)))
    return out


def positive_scores(model: BaseEstimator, X) -> tuple:
    y = model.predict(X)
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)
            s = p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()
            return y, s
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # squash to (0,1)-ish for meta features across mixed bases
        s = 1.0 / (1.0 + np.exp(-np.clip(s, -25, 25)))
        return y, s
    return y, y.astype(float)
