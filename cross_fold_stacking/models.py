"""Base and meta model constructors."""
from __future__ import annotations

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def make_base_model(name: str = "logistic", random_seed: int = 42) -> BaseEstimator:
    """
    Base model must expose decision scores; we prefer predict_proba.
    LinearSVC is wrapped with calibration for probability scores.
    """
    name = name.lower()
    if name in ("logistic", "lr", "logistic_regression"):
        return LogisticRegression(max_iter=2000, random_state=random_seed)
    if name in ("linear_svc", "svm", "linearsvc"):
        svm = LinearSVC(dual="auto", random_state=random_seed)
        return CalibratedClassifierCV(svm, cv=3)
    if name in ("extra_trees", "et", "extratrees"):
        return ExtraTreesClassifier(
            n_estimators=100, n_jobs=1, random_state=random_seed
        )
    raise ValueError(f"Unknown base_model: {name}")


def make_meta_model(name: str = "logistic", random_seed: int = 42) -> BaseEstimator:
    name = name.lower()
    if name in ("logistic", "lr", "logistic_regression"):
        return LogisticRegression(max_iter=2000, random_state=random_seed)
    if name in ("extra_trees", "et"):
        return ExtraTreesClassifier(
            n_estimators=100, n_jobs=1, random_state=random_seed
        )
    raise ValueError(f"Unknown meta_model: {name}")


def clone_model(model: BaseEstimator) -> BaseEstimator:
    return clone(model)


def predict_scores(model: BaseEstimator, X) -> tuple:
    """Return (y_pred, positive-class scores) for metrics / meta features."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    else:
        scores = y_pred.astype(float)
    return y_pred, scores
