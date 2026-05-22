"""
preprocessing.py
================
Functions for loading RNA-seq data and computing gene-level features
for ML-based filtering.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def load_expression_matrix(
    path: str,
    sep: Optional[str] = None,
    index_col: int = 0,
    log_transform: bool = True,
    cpm_normalize: bool = True,
) -> pd.DataFrame:
    """
    Load RNA-seq expression matrix from CSV/TSV file.

    Parameters
    ----------
    path : str
        Path to expression matrix (genes × samples).
    sep : str, optional
        Column separator. If None, auto-detected.
    index_col : int
        Column to use as gene index (default: 0).
    log_transform : bool
        Apply log2(CPM + 1) transformation (default: True).
    cpm_normalize : bool
        Normalize to counts per million before log transform (default: True).

    Returns
    -------
    pd.DataFrame
        Expression matrix (genes × samples), optionally log2(CPM+1) transformed.
    """
    if sep is None:
        expr = pd.read_csv(path, sep=None, engine="python", index_col=index_col)
    else:
        expr = pd.read_csv(path, sep=sep, index_col=index_col)

    # Remove non-numeric columns
    expr = expr.select_dtypes(include=[np.number])

    # Remove duplicate gene indices
    expr = expr[~expr.index.duplicated(keep="first")]

    if cpm_normalize:
        lib_sizes = expr.sum(axis=0)
        expr = expr.div(lib_sizes, axis=1) * 1e6

    if log_transform:
        expr = np.log2(expr + 1)

    return expr


def compute_gene_features(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute gene-level expression features for ML filtering.

    Parameters
    ----------
    expr : pd.DataFrame
        Log2(CPM+1) expression matrix (genes × samples).

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns:
        mean_expr, variance, cv, zero_fraction, max_expr, iqr, median_expr, skewness
    """
    features = pd.DataFrame(index=expr.index)

    features["mean_expr"]     = expr.mean(axis=1)
    features["variance"]      = expr.var(axis=1)
    features["cv"]            = expr.std(axis=1) / (expr.mean(axis=1) + 1e-8)
    features["zero_fraction"] = (expr == 0).mean(axis=1)
    features["max_expr"]      = expr.max(axis=1)
    features["iqr"]           = expr.quantile(0.75, axis=1) - expr.quantile(0.25, axis=1)
    features["median_expr"]   = expr.median(axis=1)
    features["skewness"]      = expr.apply(lambda row: skew(row), axis=1)

    return features


def split_train_test(
    features: pd.DataFrame,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Parameters
    ----------
    features : pd.DataFrame
        Gene feature matrix.
    labels : pd.Series
        Binary labels (1 = keep, 0 = filter).
    test_size : float
        Fraction of data for test set (default: 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames/Series
    """
    return train_test_split(
        features, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )


def apply_htsfilter_labels(
    features: pd.DataFrame,
    htsfilter_kept_genes: list,
) -> pd.Series:
    """
    Create binary labels based on HTSFilter output.

    Parameters
    ----------
    features : pd.DataFrame
        Gene feature matrix (index = gene names).
    htsfilter_kept_genes : list
        List of gene names retained by HTSFilter.

    Returns
    -------
    pd.Series
        Binary labels (1 = kept by HTSFilter, 0 = filtered).
    """
    labels = pd.Series(0, index=features.index, name="label")
    labels[labels.index.isin(htsfilter_kept_genes)] = 1
    return labels
