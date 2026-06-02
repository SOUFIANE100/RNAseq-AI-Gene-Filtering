"""
10_statistical_validation.py
=============================
Rigorous statistical validation of RF vs HTSFilter agreement.

Tests performed
---------------
1. McNemar test (gene-level agreement)
2. Cohen's Kappa with 95% bootstrap CI
3. Bootstrap CIs for AUC, F1, Jaccard (1,000 iterations)
4. Permutation test for Jaccard (10,000 permutations)
5. Bland-Altman analysis (mean expression agreement)
6. Lin's Concordance Correlation Coefficient (CCC)


"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import (roc_auc_score, f1_score, cohen_kappa_score,
                              confusion_matrix)
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.1)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

N_BOOT = 1000
N_PERM = 10000
SEED   = 42
rng    = np.random.default_rng(SEED)

# ── 1. Load ───────────────────────────────────────────────────────────────────
features = pd.read_csv("data/gene_features_16.csv", index_col=0)
counts   = pd.read_csv("data/counts_QC.csv", index_col=0)

y_true = features["HTSFilter_label"].values
# Use RF predictions if available, else use HTSFilter as proxy
y_pred = features["pred_RF"].values if "pred_RF" in features.columns else y_true
probs  = features["prob_RF"].values if "prob_RF" in features.columns else y_true.astype(float)

print(f"Genes: {len(y_true):,}  |  HTSFilter keep: {y_true.sum():,}  |  RF keep: {y_pred.sum():,}")

# ── 2. McNemar test ───────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
table = np.array([[tp, fp], [fn, tn]])
mcn = mcnemar(table, exact=False, correction=True)
print(f"\nMcNemar: statistic={mcn.statistic:.4f}, p={mcn.pvalue:.4e}")

# ── 3. Cohen's Kappa ──────────────────────────────────────────────────────────
kappa = cohen_kappa_score(y_true, y_pred)
kappa_boots = []
for _ in range(N_BOOT):
    idx = rng.integers(0, len(y_true), len(y_true))
    kappa_boots.append(cohen_kappa_score(y_true[idx], y_pred[idx]))
kappa_ci = np.percentile(kappa_boots, [2.5, 97.5])
print(f"Cohen's Kappa: {kappa:.4f}  95% CI [{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}]")

# ── 4. Bootstrap CIs ─────────────────────────────────────────────────────────
def jaccard(a, b):
    inter = np.sum((a == 1) & (b == 1))
    union = np.sum((a == 1) | (b == 1))
    return inter / union if union > 0 else 0.0

auc_boots, f1_boots, jac_boots = [], [], []
for _ in range(N_BOOT):
    idx = rng.integers(0, len(y_true), len(y_true))
    yt, yp, ypr = y_true[idx], y_pred[idx], probs[idx]
    if len(np.unique(yt)) < 2:
        continue
    auc_boots.append(roc_auc_score(yt, ypr))
    f1_boots.append(f1_score(yt, yp, zero_division=0))
    jac_boots.append(jaccard(yt, yp))

auc_ci = np.percentile(auc_boots, [2.5, 97.5])
f1_ci  = np.percentile(f1_boots,  [2.5, 97.5])
jac_ci = np.percentile(jac_boots, [2.5, 97.5])
print(f"AUC     : {np.mean(auc_boots):.4f}  95% CI [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
print(f"F1      : {np.mean(f1_boots):.4f}  95% CI [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
print(f"Jaccard : {np.mean(jac_boots):.4f}  95% CI [{jac_ci[0]:.4f}, {jac_ci[1]:.4f}]")

# ── 5. Permutation test ───────────────────────────────────────────────────────
obs_jac = jaccard(y_true, y_pred)
null_jac = []
for _ in range(N_PERM):
    y_shuf = rng.permutation(y_pred)
    null_jac.append(jaccard(y_true, y_shuf))
perm_p = np.mean(np.array(null_jac) >= obs_jac)
print(f"\nPermutation test: observed Jaccard={obs_jac:.4f}, null mean={np.mean(null_jac):.4f}, p={perm_p:.4e}")

# ── 6. Bland-Altman ───────────────────────────────────────────────────────────
mean_expr = np.log2(counts.mean(axis=1) + 1)
common    = mean_expr.index.intersection(features.index)
hts_expr  = mean_expr.loc[common][features.loc[common, "HTSFilter_label"] == 1]
rf_expr   = mean_expr.loc[common][features.loc[common, "pred_RF" if "pred_RF" in features.columns else "HTSFilter_label"] == 1]
common_genes = hts_expr.index.intersection(rf_expr.index)
ba_mean = (hts_expr.loc[common_genes].values + rf_expr.loc[common_genes].values) / 2
ba_diff = hts_expr.loc[common_genes].values - rf_expr.loc[common_genes].values
ba_mean_diff = np.mean(ba_diff)
ba_sd        = np.std(ba_diff)
print(f"\nBland-Altman: mean diff={ba_mean_diff:.4f}, ±1.96 SD={1.96*ba_sd:.4f}")

# ── 7. CCC ────────────────────────────────────────────────────────────────────
def ccc(x, y):
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)
    r = np.corrcoef(x, y)[0, 1]
    return 2 * r * sx * sy / (sx**2 + sy**2 + (mx - my)**2)

ccc_val = ccc(hts_expr.loc[common_genes].values, rf_expr.loc[common_genes].values)
pearson_r, pearson_p = stats.pearsonr(hts_expr.loc[common_genes].values,
                                       rf_expr.loc[common_genes].values)
print(f"CCC={ccc_val:.4f}  Pearson r={pearson_r:.4f}  p={pearson_p:.4e}")

# ── 8. Save results ───────────────────────────────────────────────────────────
results = pd.DataFrame([{
    "Metric": "AUC",           "Value": round(np.mean(auc_boots), 4),
    "CI_low": round(auc_ci[0], 4), "CI_high": round(auc_ci[1], 4)},
    {"Metric": "F1",           "Value": round(np.mean(f1_boots), 4),
    "CI_low": round(f1_ci[0], 4),  "CI_high": round(f1_ci[1], 4)},
    {"Metric": "Cohen_Kappa",  "Value": round(kappa, 4),
    "CI_low": round(kappa_ci[0], 4), "CI_high": round(kappa_ci[1], 4)},
    {"Metric": "Jaccard",      "Value": round(obs_jac, 4),
    "CI_low": round(jac_ci[0], 4),  "CI_high": round(jac_ci[1], 4)},
    {"Metric": "McNemar_p",    "Value": round(mcn.pvalue, 6), "CI_low": np.nan, "CI_high": np.nan},
    {"Metric": "Permutation_p","Value": round(perm_p, 6),     "CI_low": np.nan, "CI_high": np.nan},
    {"Metric": "CCC",          "Value": round(ccc_val, 4),    "CI_low": np.nan, "CI_high": np.nan},
    {"Metric": "Pearson_r",    "Value": round(pearson_r, 4),  "CI_low": np.nan, "CI_high": np.nan},
    {"Metric": "TP", "Value": int(tp), "CI_low": np.nan, "CI_high": np.nan},
    {"Metric": "FP", "Value": int(fp), "CI_low": np.nan, "CI_high": np.nan},
    {"Metric": "FN", "Value": int(fn), "CI_low": np.nan, "CI_high": np.nan},
    {"Metric": "TN", "Value": int(tn), "CI_low": np.nan, "CI_high": np.nan},
])
results.to_csv("results/statistical_validation_RF_vs_HTSFilter.csv", index=False)
print("\nSaved: results/statistical_validation_RF_vs_HTSFilter.csv")

# ── 9. 6-panel figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# A: Confusion matrix
ax_a = fig.add_subplot(gs[0, 0])
cm_disp = np.array([[tp, fp], [fn, tn]])
im = ax_a.imshow(cm_disp, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax_a.text(j, i, f"{cm_disp[i,j]:,}", ha="center", va="center", fontsize=11,
                   color="white" if cm_disp[i,j] > cm_disp.max()/2 else "black")
ax_a.set_xticks([0,1]); ax_a.set_yticks([0,1])
ax_a.set_xticklabels(["Pred Keep","Pred Remove"]); ax_a.set_yticklabels(["True Keep","True Remove"])
ax_a.set_title("A  Confusion Matrix", fontweight="bold")

# B: Bootstrap distributions
ax_b = fig.add_subplot(gs[0, 1])
for vals, label, color in [(auc_boots,"AUC","#0279EE"),(f1_boots,"F1","#75A025"),
                             (jac_boots,"Jaccard","#FF9400")]:
    ax_b.hist(vals, bins=30, alpha=0.6, label=f"{label} (μ={np.mean(vals):.4f})", color=color, density=True)
ax_b.set_xlabel("Metric value"); ax_b.set_ylabel("Density")
ax_b.set_title("B  Bootstrap Distributions (1,000 iter)", fontweight="bold")
ax_b.legend(fontsize=8); sns.despine(ax=ax_b)

# C: Permutation test
ax_c = fig.add_subplot(gs[0, 2])
ax_c.hist(null_jac, bins=60, color="#ECE9E2", edgecolor="grey", linewidth=0.3, density=True)
ax_c.axvline(obs_jac, color="#FF9400", lw=2.5, label=f"Observed={obs_jac:.4f}")
ax_c.set_xlabel("Jaccard similarity"); ax_c.set_ylabel("Density")
ax_c.set_title(f"C  Permutation Test (p={perm_p:.4e})", fontweight="bold")
ax_c.legend(fontsize=9); sns.despine(ax=ax_c)

# D: Bland-Altman
ax_d = fig.add_subplot(gs[1, 0])
ax_d.scatter(ba_mean, ba_diff, s=2, alpha=0.3, color="#0279EE", rasterized=True)
ax_d.axhline(ba_mean_diff, color="red", lw=2, label=f"Mean={ba_mean_diff:.4f}")
ax_d.axhline(ba_mean_diff + 1.96*ba_sd, color="grey", lw=1.5, ls="--", label=f"+1.96 SD={ba_mean_diff+1.96*ba_sd:.4f}")
ax_d.axhline(ba_mean_diff - 1.96*ba_sd, color="grey", lw=1.5, ls="--", label=f"-1.96 SD={ba_mean_diff-1.96*ba_sd:.4f}")
ax_d.set_xlabel("Mean log₂ expression"); ax_d.set_ylabel("Difference")
ax_d.set_title("D  Bland-Altman Plot", fontweight="bold")
ax_d.legend(fontsize=7); sns.despine(ax=ax_d)

# E: CCC scatter
ax_e = fig.add_subplot(gs[1, 1])
ax_e.scatter(hts_expr.loc[common_genes].values, rf_expr.loc[common_genes].values,
              s=2, alpha=0.3, color="#75A025", rasterized=True)
lim = [min(hts_expr.loc[common_genes].min(), rf_expr.loc[common_genes].min()),
       max(hts_expr.loc[common_genes].max(), rf_expr.loc[common_genes].max())]
ax_e.plot(lim, lim, 'k--', lw=1.5)
ax_e.set_xlabel("HTSFilter log₂ mean expression"); ax_e.set_ylabel("RF log₂ mean expression")
ax_e.set_title(f"E  CCC={ccc_val:.4f}  r={pearson_r:.4f}", fontweight="bold")
sns.despine(ax=ax_e)

# F: Summary table
ax_f = fig.add_subplot(gs[1, 2])
ax_f.axis("off")
tbl_data = [["Metric","Value","95% CI"],
             ["AUC",    f"{np.mean(auc_boots):.4f}", f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]"],
             ["F1",     f"{np.mean(f1_boots):.4f}",  f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]"],
             ["Kappa",  f"{kappa:.4f}",               f"[{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}]"],
             ["Jaccard",f"{obs_jac:.4f}",             f"[{jac_ci[0]:.4f}, {jac_ci[1]:.4f}]"],
             ["McNemar p", f"{mcn.pvalue:.2e}", "—"],
             ["Perm. p",   f"{perm_p:.2e}",    "—"],
             ["CCC",       f"{ccc_val:.4f}",   "—"],
             ["Pearson r", f"{pearson_r:.4f}", "—"],
             ["TP/FP/FN/TN", f"{tp}/{fp}/{fn}/{tn}", "—"]]
tbl = ax_f.table(cellText=tbl_data[1:], colLabels=tbl_data[0],
                  loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1.2, 1.4)
ax_f.set_title("F  Summary Statistics", fontweight="bold")

plt.suptitle("Statistical Validation: RF vs HTSFilter", fontsize=14, fontweight="bold", y=1.01)
plt.savefig("figures/statistical_validation_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/statistical_validation_overview.png")
print("\nDone.")
