"""
03_model_training_SVM_RF_KNN_MLP.py
====================================
Train SVM, Random Forest, KNN, and MLP classifiers on the 16-feature
gene-expression matrix. Evaluate with stratified 5-fold cross-validation.

Hyperparameters
---------------
SVM  : kernel=rbf, C=10, gamma=scale, probability=True
RF   : n_estimators=500, n_jobs=-1
KNN  : n_neighbors=5, n_jobs=-1
MLP  : hidden_layer_sizes=(256,128,64), max_iter=500, early_stopping=True

Inputs
------
- data/gene_features_16.csv

Outputs
-------
- models/SVM.pkl, RF.pkl, KNN.pkl, MLP.pkl
- results/cv_metrics_SVM_RF_KNN_MLP.csv
- figures/roc_curves_SVM_RF_KNN_MLP.png
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", font_scale=1.2)
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

# ── 1. Load features ──────────────────────────────────────────────────────────
feat_cols = ['mean','median','std','IQR','min','max','range','p10','p25','p75',
             'p90','frac_expr','skewness','kurtosis','CV','log2_mean']
df = pd.read_csv("data/gene_features_16.csv", index_col=0)
df[feat_cols] = df[feat_cols].fillna(0)
X = df[feat_cols].values
y = df["HTSFilter_label"].values
print(f"Dataset: {X.shape[0]:,} genes, {X.shape[1]} features")
print(f"Label balance: {y.sum():,} keep / {(y==0).sum():,} remove")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 2. Cross-validation ───────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "SVM":  (SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42), X_scaled),
    "RF":   (RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42), X),
    "KNN":  (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), X_scaled),
    "MLP":  (MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=500,
                            early_stopping=True, random_state=42), X_scaled),
}

records = []
fig, ax = plt.subplots(figsize=(7, 6))
colors = {"SVM": "#0279EE", "RF": "#75A025", "KNN": "#FF9400", "MLP": "#E9ED4C"}

for name, (model, X_use) in models.items():
    print(f"  CV {name} …", flush=True)
    probs = cross_val_predict(model, X_use, y, cv=cv, method="predict_proba")[:, 1]
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
    # Fit on full data and save
    model.fit(X_use, y)
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"    AUC={auc:.4f}  F1={records[-1]['F1']:.4f}")

ax.plot([0,1],[0,1],'k--',lw=1,label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — SVM / RF / KNN / MLP (5-fold CV)")
ax.legend(fontsize=10); sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("figures/roc_curves_SVM_RF_KNN_MLP.png", dpi=150, bbox_inches="tight")
plt.close()

cv_df = pd.DataFrame(records)
cv_df.to_csv("results/cv_metrics_SVM_RF_KNN_MLP.csv", index=False)
print("\nCV metrics:")
print(cv_df.to_string(index=False))
print("\nSaved: results/cv_metrics_SVM_RF_KNN_MLP.csv")
print("Saved: figures/roc_curves_SVM_RF_KNN_MLP.png")
print("Saved: models/{SVM,RF,KNN,MLP}.pkl")
print("\nDone.")
