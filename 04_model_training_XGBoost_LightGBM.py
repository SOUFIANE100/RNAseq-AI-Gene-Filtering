"""
04_model_training_XGBoost_LightGBM.py
======================================
Train XGBoost and LightGBM gradient-boosting classifiers.
Evaluate with stratified 5-fold cross-validation.

Hyperparameters
---------------
XGBoost  : n_estimators=300, max_depth=6, learning_rate=0.1,
           eval_metric='logloss', n_jobs=-1
LightGBM : n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1

Inputs
------
- data/gene_features_16.csv

Outputs
-------
- models/XGBoost.pkl, LightGBM.pkl
- results/cv_metrics_XGBoost_LightGBM.csv
- figures/roc_curves_XGBoost_LightGBM.png
- figures/feature_importance_XGBoost_LightGBM.png
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", font_scale=1.2)
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

feat_cols = ['mean','median','std','IQR','min','max','range','p10','p25','p75',
             'p90','frac_expr','skewness','kurtosis','CV','log2_mean']

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/gene_features_16.csv", index_col=0)
df[feat_cols] = df[feat_cols].fillna(0)
X = df[feat_cols].values
y = df["HTSFilter_label"].values
print(f"Dataset: {X.shape[0]:,} genes, {X.shape[1]} features")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "XGBoost":  XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               eval_metric='logloss', n_jobs=-1,
                               random_state=42, verbosity=0),
    "LightGBM": LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                n_jobs=-1, random_state=42, verbose=-1),
}

# ── 2. Cross-validation ───────────────────────────────────────────────────────
records = []
fig, ax = plt.subplots(figsize=(7, 6))
colors = {"XGBoost": "#FD9BED", "LightGBM": "#000000"}

for name, model in models.items():
    print(f"  CV {name} …", flush=True)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=colors[name], lw=2)
    records.append({
        "Model": name,
        "AUC":       round(auc, 4),
        "F1":        round(f1_score(y, preds), 4),
        "Accuracy":  round(accuracy_score(y, preds), 4),
        "Precision": round(precision_score(y, preds), 4),
        "Recall":    round(recall_score(y, preds), 4),
    })
    model.fit(X, y)
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"    AUC={auc:.4f}  F1={records[-1]['F1']:.4f}")

ax.plot([0,1],[0,1],'k--',lw=1,label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — XGBoost / LightGBM (5-fold CV)")
ax.legend(fontsize=10); sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("figures/roc_curves_XGBoost_LightGBM.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Feature importance ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, model) in zip(axes, models.items()):
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=True)
    imp.plot.barh(ax=ax, color="#0279EE", edgecolor="white")
    ax.set_title(f"{name} — Feature Importance")
    ax.set_xlabel("Importance score")
    sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("figures/feature_importance_XGBoost_LightGBM.png", dpi=150, bbox_inches="tight")
plt.close()

cv_df = pd.DataFrame(records)
cv_df.to_csv("results/cv_metrics_XGBoost_LightGBM.csv", index=False)
print("\nCV metrics:")
print(cv_df.to_string(index=False))
print("\nSaved: results/cv_metrics_XGBoost_LightGBM.csv")
print("Saved: figures/roc_curves_XGBoost_LightGBM.png")
print("Saved: figures/feature_importance_XGBoost_LightGBM.png")
print("Saved: models/{XGBoost,LightGBM}.pkl")
print("\nDone.")
