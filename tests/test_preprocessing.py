"""
Unit tests for preprocessing module.
Run with: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brca_ml_filtering.preprocessing import compute_gene_features, split_train_test


@pytest.fixture
def mock_expr():
    """Create a small mock expression matrix (50 genes × 20 samples)."""
    np.random.seed(42)
    data = np.random.negative_binomial(5, 0.3, size=(50, 20)).astype(float)
    genes   = [f"GENE{i}" for i in range(50)]
    samples = [f"SAMPLE{i}" for i in range(20)]
    return pd.DataFrame(data, index=genes, columns=samples)


def test_compute_gene_features_shape(mock_expr):
    features = compute_gene_features(mock_expr)
    assert features.shape == (50, 8), f"Expected (50, 8), got {features.shape}"


def test_compute_gene_features_columns(mock_expr):
    features = compute_gene_features(mock_expr)
    expected_cols = ["mean_expr", "variance", "cv", "zero_fraction",
                     "max_expr", "iqr", "median_expr", "skewness"]
    assert list(features.columns) == expected_cols


def test_compute_gene_features_no_nan(mock_expr):
    features = compute_gene_features(mock_expr)
    assert not features.isnull().any().any(), "Features contain NaN values"


def test_compute_gene_features_cv_nonnegative(mock_expr):
    features = compute_gene_features(mock_expr)
    assert (features["cv"] >= 0).all(), "CV should be non-negative"


def test_split_train_test_sizes(mock_expr):
    features = compute_gene_features(mock_expr)
    labels = pd.Series(np.random.randint(0, 2, 50), index=features.index)
    X_train, X_test, y_train, y_test = split_train_test(features, labels, test_size=0.2)
    assert len(X_train) == 40
    assert len(X_test)  == 10
    assert len(y_train) == 40
    assert len(y_test)  == 10


def test_split_train_test_no_overlap(mock_expr):
    features = compute_gene_features(mock_expr)
    labels = pd.Series(np.random.randint(0, 2, 50), index=features.index)
    X_train, X_test, _, _ = split_train_test(features, labels)
    assert len(set(X_train.index) & set(X_test.index)) == 0, "Train/test overlap detected"
