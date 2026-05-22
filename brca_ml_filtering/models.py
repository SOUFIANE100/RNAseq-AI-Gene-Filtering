"""
models.py
=========
ML model definitions, training, and prediction for gene filtering.
Supports: LightGBM, XGBoost, RF, MLP, SVM, KNN, CNN, RNN-LSTM.
"""

import numpy as np
import pandas as pd
import pickle
import time
from typing import Dict, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ── Default hyperparameters (tuned on TCGA BRCA) ─────────────────────────────

DEFAULT_PARAMS = {
    "LightGBM": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    },
    "RF": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "MLP": {
        "hidden_layer_sizes": (256, 128, 64),
        "activation": "relu",
        "max_iter": 100,
        "random_state": 42,
        "early_stopping": True,
    },
    "SVM": {
        "kernel": "rbf",
        "C": 10,
        "gamma": 0.01,
        "probability": True,
        "random_state": 42,
    },
    "KNN": {
        "n_neighbors": 5,
        "metric": "euclidean",
        "n_jobs": -1,
    },
}


def build_model(model_name: str, params: Optional[dict] = None):
    """
    Build a scikit-learn compatible model.

    Parameters
    ----------
    model_name : str
        One of: LightGBM, XGBoost, RF, MLP, SVM, KNN
    params : dict, optional
        Override default hyperparameters.

    Returns
    -------
    sklearn estimator or Pipeline
    """
    p = {**DEFAULT_PARAMS.get(model_name, {}), **(params or {})}

    if model_name == "LightGBM":
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        return lgb.LGBMClassifier(**p)

    elif model_name == "XGBoost":
        if not HAS_XGB:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        return xgb.XGBClassifier(**p)

    elif model_name == "RF":
        return RandomForestClassifier(**p)

    elif model_name == "MLP":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(**p)),
        ])

    elif model_name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(**p)),
        ])

    elif model_name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(**p)),
        ])

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: LightGBM, XGBoost, RF, MLP, SVM, KNN")


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_names: Optional[list] = None,
    params: Optional[Dict[str, dict]] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Train all specified ML models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels.
    model_names : list, optional
        Models to train. Default: all 6 sklearn models.
    params : dict, optional
        Per-model hyperparameter overrides.
    verbose : bool
        Print training progress.

    Returns
    -------
    dict
        {model_name: fitted_model}
    """
    if model_names is None:
        model_names = ["LightGBM", "XGBoost", "RF", "MLP", "SVM", "KNN"]

    trained_models = {}
    for name in model_names:
        if verbose:
            print(f"Training {name}...", end=" ", flush=True)
        t0 = time.time()
        model = build_model(name, (params or {}).get(name))
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        trained_models[name] = model
        if verbose:
            print(f"done ({elapsed:.1f}s)")

    return trained_models


def predict_filtering(
    model,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict gene filtering labels and probabilities.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        Decision threshold for binary classification (default: 0.5).

    Returns
    -------
    labels : np.ndarray
        Binary predictions (1 = keep, 0 = filter).
    proba : np.ndarray
        Probability of "keep" class.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.decision_function(X)
        proba = (proba - proba.min()) / (proba.max() - proba.min())

    labels = (proba >= threshold).astype(int)
    return labels, proba


def save_models(models: dict, path: str) -> None:
    """Save trained models to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(models, f)
    print(f"Models saved to {path}")


def load_models(path: str) -> dict:
    """Load trained models from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
