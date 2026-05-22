"""
05_model_training_CNN_RNN.py
============================
Train 1-D CNN and bidirectional LSTM classifiers using PyTorch.
Evaluate with stratified 5-fold cross-validation.

Architecture
------------
CNN  : Conv1d(1→32, k=3) → BN → ReLU → Conv1d(32→64, k=3) → BN → ReLU
       → AdaptiveAvgPool1d(4) → Linear(256→128) → ReLU → Dropout(0.3)
       → Linear(128→64) → ReLU → Linear(64→1)

RNN  : BiLSTM(1→64, 2 layers, dropout=0.2) → Linear(128→64) → ReLU
       → Dropout(0.3) → Linear(64→1)

Training
--------
Optimizer : Adam (lr=1e-3, weight_decay=1e-4)
Scheduler : CosineAnnealingLR (T_max=50)
Loss      : BCEWithLogitsLoss (pos_weight for class imbalance)
Epochs    : 50
Batch     : 512

Inputs
------
- data/gene_features_16.csv

Outputs
-------
- models/CNN.pt, RNN.pt
- results/cv_metrics_CNN_RNN.csv
- figures/roc_curves_CNN_RNN.png
- figures/training_loss_CNN_RNN.png
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.2)
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS  = 50
BATCH   = 512
LR      = 1e-3
SEED    = 42
torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

feat_cols = ['mean','median','std','IQR','min','max','range','p10','p25','p75',
             'p90','frac_expr','skewness','kurtosis','CV','log2_mean']

# ── 1. Load & scale ───────────────────────────────────────────────────────────
df = pd.read_csv("data/gene_features_16.csv", index_col=0)
df[feat_cols] = df[feat_cols].fillna(0)
X_raw = df[feat_cols].values
y     = df["HTSFilter_label"].values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f"Dataset: {X.shape[0]:,} genes, {X.shape[1]} features")

# ── 2. Model definitions ──────────────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self, n_feat=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).view(x.size(0), -1)).squeeze(1)

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(2))
        return self.fc(out[:, -1, :]).squeeze(1)

# ── 3. Training helper ────────────────────────────────────────────────────────
def train_model(ModelClass, X_tr, y_tr, n_epochs=EPOCHS, batch_size=BATCH):
    X_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_t = torch.FloatTensor(y_tr).to(DEVICE)
    pos_w = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()]).to(DEVICE)
    model = ModelClass().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    losses = []
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        losses.append(epoch_loss / len(loader))
    return model, losses

def predict_proba(model, X_np):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_np).to(DEVICE)
        return torch.sigmoid(model(X_t)).cpu().numpy()

# ── 4. 5-fold CV ──────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
records = []
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
colors = {"CNN": "#0279EE", "RNN": "#FF9400"}

for model_name, ModelClass in [("CNN", CNN1D), ("RNN", RNNModel)]:
    print(f"\n  CV {model_name} …")
    all_probs = np.zeros(len(y))
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        m, _ = train_model(ModelClass, X[tr_idx], y[tr_idx])
        all_probs[val_idx] = predict_proba(m, X[val_idx])
        print(f"    Fold {fold+1}/5 done", flush=True)
    preds = (all_probs >= 0.5).astype(int)
    auc = roc_auc_score(y, all_probs)
    fpr, tpr, _ = roc_curve(y, all_probs)
    ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})",
                color=colors[model_name], lw=2)
    records.append({
        "Model": model_name,
        "AUC":       round(auc, 4),
        "F1":        round(f1_score(y, preds), 4),
        "Accuracy":  round(accuracy_score(y, preds), 4),
        "Precision": round(precision_score(y, preds), 4),
        "Recall":    round(recall_score(y, preds), 4),
    })
    print(f"    AUC={auc:.4f}  F1={records[-1]['F1']:.4f}")

ax_roc.plot([0,1],[0,1],'k--',lw=1,label="Random")
ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves — CNN / RNN (5-fold CV)")
ax_roc.legend(fontsize=10); sns.despine(ax=ax_roc)
plt.tight_layout()
plt.savefig("figures/roc_curves_CNN_RNN.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5. Train final models on full data ───────────────────────────────────────
print("\nTraining final models on full data …")
for model_name, ModelClass in [("CNN", CNN1D), ("RNN", RNNModel)]:
    m, losses = train_model(ModelClass, X, y)
    torch.save(m.state_dict(), f"models/{model_name}.pt")
    print(f"  Saved: models/{model_name}.pt  (final loss={losses[-1]:.4f})")

# ── 6. Training loss figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (model_name, ModelClass) in zip(axes, [("CNN", CNN1D), ("RNN", RNNModel)]):
    _, losses = train_model(ModelClass, X, y)
    ax.plot(losses, color=colors[model_name], lw=2)
    ax.set_title(f"{model_name} — Training Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("figures/training_loss_CNN_RNN.png", dpi=150, bbox_inches="tight")
plt.close()

cv_df = pd.DataFrame(records)
cv_df.to_csv("results/cv_metrics_CNN_RNN.csv", index=False)
print("\nCV metrics:")
print(cv_df.to_string(index=False))
print("\nSaved: results/cv_metrics_CNN_RNN.csv")
print("Saved: figures/roc_curves_CNN_RNN.png")
print("Saved: figures/training_loss_CNN_RNN.png")
print("\nDone.")
