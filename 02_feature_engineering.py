"""
02_feature_engineering.py
=========================
Compute 16 per-gene statistical features from the QC-passed count matrix.

Features
--------
mean, median, std, IQR, min, max, range, p10, p25, p75, p90,
frac_expr (fraction of samples with count > 0),
skewness, kurtosis, CV (coefficient of variation), log2_mean

Inputs
------
- data/counts_QC.csv   : QC-passed count matrix (genes × samples)
- data/labels_QC.csv   : HTSFilter binary labels

Outputs
-------
- data/gene_features_16.csv : feature matrix with HTSFilter label column
- figures/feature_distributions.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.1)
Path("figures").mkdir(exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading QC data …")
counts = pd.read_csv("data/counts_QC.csv", index_col=0)
labels = pd.read_csv("data/labels_QC.csv", index_col=0)
y = labels.iloc[:, 0].values
print(f"  {counts.shape[0]:,} genes × {counts.shape[1]:,} samples")

# ── 2. Compute features ───────────────────────────────────────────────────────
print("Computing 16 features …")
X = counts.values.astype(float)

feat = pd.DataFrame(index=counts.index)
feat["mean"]      = X.mean(axis=1)
feat["median"]    = np.median(X, axis=1)
feat["std"]       = X.std(axis=1)
feat["IQR"]       = np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1)
feat["min"]       = X.min(axis=1)
feat["max"]       = X.max(axis=1)
feat["range"]     = feat["max"] - feat["min"]
feat["p10"]       = np.percentile(X, 10, axis=1)
feat["p25"]       = np.percentile(X, 25, axis=1)
feat["p75"]       = np.percentile(X, 75, axis=1)
feat["p90"]       = np.percentile(X, 90, axis=1)
feat["frac_expr"] = (X > 0).mean(axis=1)
feat["skewness"]  = stats.skew(X, axis=1)
feat["kurtosis"]  = stats.kurtosis(X, axis=1)
feat["CV"]        = np.where(feat["mean"] > 0, feat["std"] / feat["mean"], 0)
feat["log2_mean"] = np.log2(feat["mean"] + 1)
feat["HTSFilter_label"] = y
feat["gene_id"]   = counts.index

print(f"  Feature matrix: {feat.shape}")
print(f"  Missing values: {feat.iloc[:, :16].isna().sum().sum()}")

# ── 3. Feature distributions ─────────────────────────────────────────────────
feat_cols = ['mean','median','std','IQR','frac_expr','skewness',
             'kurtosis','CV','log2_mean','p90']
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()
palette = {0: "#FF9400", 1: "#75A025"}
for i, col in enumerate(feat_cols):
    for lbl, color in palette.items():
        subset = feat.loc[feat["HTSFilter_label"] == lbl, col].dropna()
        axes[i].hist(subset, bins=50, alpha=0.6, color=color,
                     label=f"{'Keep' if lbl==1 else 'Remove'}", density=True)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    sns.despine(ax=axes[i])
axes[0].legend(fontsize=9)
plt.suptitle("Feature distributions by HTSFilter label", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/feature_distributions.png")

# ── 4. Save ───────────────────────────────────────────────────────────────────
feat.to_csv("data/gene_features_16.csv")
print("  Saved: data/gene_features_16.csv")
print("\nDone.")
