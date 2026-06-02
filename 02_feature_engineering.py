#!/usr/bin/env python3
"""
02_feature_engineering.py
=========================
Compute 18 per-gene statistical features, perform ANOVA feature ranking,
and generate the combined ANOVA + correlation figure.

Inputs
------
- data/counts_QC.csv : count matrix, genes × samples
- data/labels_QC.csv : binary labels, 1 = retained, 0 = filtered

Outputs
-------
- data/gene_features_18.csv
- data/feature_importance_anova.csv
- figures/feature_anova_correlation_18.png
- figures/feature_anova_correlation_18.pdf
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Settings ────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", font_scale=1.1)

Path("data").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

# ── 1. Load data ────────────────────────────────────────────────────────────
print("Loading QC data ...")

counts = pd.read_csv("data/counts_QC.csv", index_col=0)
labels = pd.read_csv("data/labels_QC.csv", index_col=0)

y = labels.iloc[:, 0].values

print(f"Counts matrix: {counts.shape[0]:,} genes × {counts.shape[1]:,} samples")

X = counts.values.astype(float)

# ── 2. Compute 18 statistical features ──────────────────────────────────────
print("Computing 18 statistical features ...")

feat = pd.DataFrame(index=counts.index)

feat["mean_expr"] = X.mean(axis=1)
feat["var_expr"] = X.var(axis=1)
feat["std_expr"] = X.std(axis=1)
feat["median_expr"] = np.median(X, axis=1)

feat["cv_expr"] = np.where(
    feat["mean_expr"] > 0,
    feat["std_expr"] / feat["mean_expr"],
    0
)

feat["sparsity"] = (X == 0).mean(axis=1)

feat["q25"] = np.percentile(X, 25, axis=1)
feat["q75"] = np.percentile(X, 75, axis=1)
feat["iqr_expr"] = feat["q75"] - feat["q25"]

feat["min_expr"] = X.min(axis=1)
feat["max_expr"] = X.max(axis=1)
feat["range_expr"] = feat["max_expr"] - feat["min_expr"]

feat["skewness"] = stats.skew(X, axis=1, nan_policy="omit")
feat["kurtosis"] = stats.kurtosis(X, axis=1, nan_policy="omit")

nonzero_count = (X > 0).sum(axis=1)
nonzero_sum = X.sum(axis=1)

feat["mean_nonzero"] = np.where(
    nonzero_count > 0,
    nonzero_sum / nonzero_count,
    0
)

feat["log_mean"] = np.log1p(feat["mean_expr"])
feat["log_var"] = np.log1p(feat["var_expr"])

feat["pct_above_mean"] = (
    X > feat["mean_expr"].values[:, None]
).mean(axis=1)

feature_cols = [
    "mean_expr", "var_expr", "std_expr", "median_expr", "cv_expr",
    "sparsity", "q25", "q75", "iqr_expr",
    "min_expr", "max_expr", "range_expr",
    "skewness", "kurtosis",
    "mean_nonzero",
    "log_mean", "log_var", "pct_above_mean"
]

feat[feature_cols] = feat[feature_cols].replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

feat["target"] = y
feat["filter_status"] = np.where(y == 1, "retained", "filtered")
feat["gene_id"] = counts.index

print(f"Feature matrix: {feat.shape}")
print(f"Missing values: {feat[feature_cols].isna().sum().sum()}")

# ── 3. ANOVA feature importance ─────────────────────────────────────────────
print("Running ANOVA feature importance ...")

display_names = {
    "mean_expr": "Mean Expr",
    "var_expr": "Variance",
    "std_expr": "Std Dev",
    "median_expr": "Median",
    "cv_expr": "CV",
    "sparsity": "Sparsity",
    "q25": "Q25",
    "q75": "Q75",
    "iqr_expr": "IQR",
    "min_expr": "Min",
    "max_expr": "Max",
    "range_expr": "Range",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis",
    "mean_nonzero": "Mean (non-zero)",
    "log_mean": "Log Mean",
    "log_var": "Log Var",
    "pct_above_mean": "% Above Mean"
}

anova_rows = []

for col in feature_cols:
    retained = feat.loc[feat["target"] == 1, col]
    filtered = feat.loc[feat["target"] == 0, col]

    F, p = f_oneway(retained, filtered)

    anova_rows.append({
        "Feature": display_names[col],
        "feature_col": col,
        "F_score": F,
        "p_value": p
    })

df_anova = pd.DataFrame(anova_rows).sort_values(
    "F_score",
    ascending=False
)

df_anova.to_csv(
    "data/feature_importance_anova.csv",
    index=False
)

print("Top ANOVA features:")
print(df_anova.head(10).to_string(index=False))

# ── 4. Combined figure: ANOVA + correlation matrix ──────────────────────────
print("Creating combined ANOVA + correlation figure ...")

df_plot = df_anova.copy()
df_plot = df_plot.sort_values("F_score", ascending=True)

median_f = df_plot["F_score"].median()

corr_df = feat[feature_cols].rename(columns=display_names)
corr_matrix = corr_df.corr(method="pearson")

fig, axes = plt.subplots(
    1, 2,
    figsize=(20, 8),
    gridspec_kw={"width_ratios": [1.05, 1.45]}
)

# Panel A — ANOVA
bar_colors = [
    "#2a9d8f" if value >= median_f else "#e9c46a"
    for value in df_plot["F_score"]
]

axes[0].barh(
    df_plot["Feature"],
    df_plot["F_score"],
    color=bar_colors,
    edgecolor="white"
)

axes[0].axvline(
    median_f,
    color="red",
    linestyle="--",
    linewidth=1,
    label="Median F-score"
)

axes[0].set_xlabel("ANOVA F-Score (Kept vs Filtered)")
axes[0].set_ylabel("")
axes[0].set_title(
    "Feature Discriminative Power\n(ANOVA F-score ranking)",
    fontsize=12,
    fontweight="bold"
)

axes[0].legend(
    frameon=False,
    fontsize=8,
    loc="lower right"
)

axes[0].grid(
    axis="x",
    linestyle="--",
    alpha=0.25
)

sns.despine(ax=axes[0])

axes[0].text(
    -0.18,
    1.03,
    "A",
    transform=axes[0].transAxes,
    fontsize=28,
    fontweight="bold"
)

# Panel B — Correlation matrix
sns.heatmap(
    corr_matrix,
    ax=axes[1],
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 6},
    square=True,
    linewidths=0.3,
    linecolor="white",
    cbar_kws={
        "label": "Pearson r",
        "shrink": 0.75
    }
)

axes[1].set_title(
    "Feature Correlation Matrix (18 per-gene features)",
    fontsize=12,
    fontweight="bold"
)

axes[1].tick_params(axis="x", labelrotation=90, labelsize=7)
axes[1].tick_params(axis="y", labelsize=7)

axes[1].text(
    -0.12,
    1.03,
    "B",
    transform=axes[1].transAxes,
    fontsize=28,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "figures/feature_anova_correlation_18.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "figures/feature_anova_correlation_18.pdf",
    bbox_inches="tight"
)

plt.close()

print("Saved: figures/feature_anova_correlation_18.png")
print("Saved: figures/feature_anova_correlation_18.pdf")

# ── 5. Save final feature matrix ────────────────────────────────────────────
feat.to_csv(
    "data/gene_features_18.csv",
    index=False
)

print("Saved: data/gene_features_18.csv")
print("Saved: data/feature_importance_anova.csv")
print("Done.")
