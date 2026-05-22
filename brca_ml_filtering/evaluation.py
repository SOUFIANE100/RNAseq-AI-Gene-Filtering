"""
evaluation.py
=============
Model evaluation metrics: classification performance, filtering agreement,
and PCA-based quality assessment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    cohen_kappa_score,
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, Optional


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_proba : array-like, optional
        Predicted probabilities for AUC computation.

    Returns
    -------
    dict
        accuracy, precision, recall, f1, auc (if proba provided),
        TN, FP, FN, TP
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard index between two gene sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_filtering_agreement(
    ml_kept: set,
    ref_kept: set,
    all_genes: set,
) -> Dict[str, float]:
    """
    Compute agreement metrics between ML filtering and reference method.

    Parameters
    ----------
    ml_kept : set
        Genes retained by ML model.
    ref_kept : set
        Genes retained by reference method (e.g., HTSFilter).
    all_genes : set
        All genes in the dataset.

    Returns
    -------
    dict
        jaccard, kappa, f1, accuracy, n_both, n_only_ml, n_only_ref, n_neither
    """
    both    = ml_kept & ref_kept
    only_ml = ml_kept - ref_kept
    only_ref = ref_kept - ml_kept
    neither = all_genes - (ml_kept | ref_kept)

    # Binary labels for kappa/f1/accuracy
    y_ref = np.array([1 if g in ref_kept else 0 for g in all_genes])
    y_ml  = np.array([1 if g in ml_kept  else 0 for g in all_genes])

    return {
        "jaccard":    compute_jaccard(ml_kept, ref_kept),
        "kappa":      cohen_kappa_score(y_ref, y_ml),
        "f1":         f1_score(y_ref, y_ml, zero_division=0),
        "accuracy":   accuracy_score(y_ref, y_ml),
        "n_both":     len(both),
        "n_only_ml":  len(only_ml),
        "n_only_ref": len(only_ref),
        "n_neither":  len(neither),
    }


def compute_silhouette(
    expr: pd.DataFrame,
    labels: pd.Series,
    n_components: int = 2,
    random_state: int = 42,
) -> float:
    """
    Compute PCA silhouette score for sample separation.

    Parameters
    ----------
    expr : pd.DataFrame
        Expression matrix (genes × samples). Will be transposed for PCA.
    labels : pd.Series
        Sample group labels (e.g., Disease/Normal).
    n_components : int
        Number of PCA components (default: 2).
    random_state : int
        Random seed.

    Returns
    -------
    float
        Silhouette score on PC1–PC2 coordinates.
    """
    X = expr.T.values  # samples × genes
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)

    # Encode labels as integers
    unique_labels = labels.unique()
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels])

    return silhouette_score(coords, y)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for AUC.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities.
    n_bootstrap : int
        Number of bootstrap replicates.
    ci : float
        Confidence level (default: 0.95).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        auc_mean, auc_lower, auc_upper
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))

    alpha = (1 - ci) / 2
    return {
        "auc_mean":  np.mean(aucs),
        "auc_lower": np.percentile(aucs, 100 * alpha),
        "auc_upper": np.percentile(aucs, 100 * (1 - alpha)),
    }
