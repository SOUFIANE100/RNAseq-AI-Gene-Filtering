"""
Unit tests for evaluation module.
Run with: pytest tests/
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brca_ml_filtering.evaluation import (
    evaluate_model, compute_jaccard, compute_filtering_agreement, bootstrap_auc_ci
)


def test_evaluate_model_perfect():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 1])
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["accuracy"]  == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"]    == 1.0
    assert metrics["f1"]        == 1.0
    assert metrics["TP"] == 3
    assert metrics["TN"] == 2
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0


def test_evaluate_model_with_proba():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 1])
    y_proba = np.array([0.9, 0.8, 0.1, 0.2, 0.95])
    metrics = evaluate_model(y_true, y_pred, y_proba)
    assert "auc" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0


def test_jaccard_identical():
    a = {1, 2, 3, 4}
    assert compute_jaccard(a, a) == 1.0


def test_jaccard_disjoint():
    a = {1, 2}
    b = {3, 4}
    assert compute_jaccard(a, b) == 0.0


def test_jaccard_partial():
    a = {1, 2, 3}
    b = {2, 3, 4}
    j = compute_jaccard(a, b)
    assert abs(j - 2/4) < 1e-10


def test_filtering_agreement_perfect():
    genes = {"A", "B", "C", "D", "E"}
    kept  = {"A", "B", "C"}
    result = compute_filtering_agreement(kept, kept, genes)
    assert result["jaccard"]  == 1.0
    assert result["kappa"]    == 1.0
    assert result["n_both"]   == 3
    assert result["n_only_ml"] == 0
    assert result["n_only_ref"] == 0


def test_bootstrap_auc_ci_shape():
    np.random.seed(42)
    y_true  = np.random.randint(0, 2, 100)
    y_proba = np.random.uniform(0, 1, 100)
    ci = bootstrap_auc_ci(y_true, y_proba, n_bootstrap=100)
    assert "auc_mean"  in ci
    assert "auc_lower" in ci
    assert "auc_upper" in ci
    assert ci["auc_lower"] <= ci["auc_mean"] <= ci["auc_upper"]
