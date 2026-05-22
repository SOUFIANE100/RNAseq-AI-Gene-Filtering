"""
06_shap_interpretability.py
============================
Compute SHAP values for the best-performing models (RF and XGBoost)
to identify the most important gene-expression features.

Inputs
------
- data/gene_features_16.csv
- models/RF.pkl
- models/XGBoost.pkl

Outputs
-------
- results/shap_feature_importance.csv
- figures/shap_summary_RF.png
- figures/shap_summary_XGBoost.png
- figures/shap_beeswarm_RF.png
"""

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.2)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

feat_cols = ['mean','median','std','IQR','min','max','range','p10','p25','p75',
             'p90','frac_expr','skewness','kurtosis','CV','log2_mean']

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/gene_features_16.csv", index_col=0)
df[feat_cols] = df[feat_cols].fillna(0)
X = df[feat_cols].values
y = df["HTSFilter_label"].values

with open("models/RF.pkl", "rb") as f:
    rf = pickle.load(f)
with open("models/XGBoost.pkl", "rb") as f:
    xgb = pickle.load(f)

# ── 2. SHAP for RF ────────────────────────────────────────────────────────────
print("Computing SHAP values for RF …")
# Use a background sample for speed
bg_idx = np.random.RandomState(42).choice(len(X), size=500, replace=False)
explainer_rf = shap.TreeExplainer(rf)
shap_rf = explainer_rf.shap_values(X[bg_idx])
# shap_rf is a list [class0, class1]; take class1
if isinstance(shap_rf, list):
    shap_rf = shap_rf[1]

# Bar plot
mean_abs_rf = np.abs(shap_rf).mean(axis=0)
imp_rf = pd.Series(mean_abs_rf, index=feat_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
imp_rf.plot.barh(ax=ax, color="#75A025", edgecolor="white")
ax.set_title("RF — Mean |SHAP| Feature Importance")
ax.set_xlabel("Mean |SHAP value|")
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("figures/shap_summary_RF.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/shap_summary_RF.png")

# Beeswarm
shap.summary_plot(shap_rf, X[bg_idx], feature_names=feat_cols, show=False)
plt.tight_layout()
plt.savefig("figures/shap_beeswarm_RF.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/shap_beeswarm_RF.png")

# ── 3. SHAP for XGBoost ───────────────────────────────────────────────────────
print("Computing SHAP values for XGBoost …")
explainer_xgb = shap.TreeExplainer(xgb)
shap_xgb = explainer_xgb.shap_values(X[bg_idx])
if isinstance(shap_xgb, list):
    shap_xgb = shap_xgb[1]

mean_abs_xgb = np.abs(shap_xgb).mean(axis=0)
imp_xgb = pd.Series(mean_abs_xgb, index=feat_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
imp_xgb.plot.barh(ax=ax, color="#FD9BED", edgecolor="white")
ax.set_title("XGBoost — Mean |SHAP| Feature Importance")
ax.set_xlabel("Mean |SHAP value|")
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("figures/shap_summary_XGBoost.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/shap_summary_XGBoost.png")

# ── 4. Save combined importance table ────────────────────────────────────────
shap_df = pd.DataFrame({
    "Feature":          feat_cols,
    "RF_mean_abs_SHAP": mean_abs_rf,
    "XGB_mean_abs_SHAP": mean_abs_xgb,
    "RF_rank":          pd.Series(mean_abs_rf).rank(ascending=False).astype(int).values,
    "XGB_rank":         pd.Series(mean_abs_xgb).rank(ascending=False).astype(int).values,
}).sort_values("RF_mean_abs_SHAP", ascending=False)
shap_df.to_csv("results/shap_feature_importance.csv", index=False)
print("\nTop 5 features (RF SHAP):")
print(shap_df.head(5).to_string(index=False))
print("\nSaved: results/shap_feature_importance.csv")
print("\nDone.")
