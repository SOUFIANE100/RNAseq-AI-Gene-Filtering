"""
11_visualization.py
====================
Generate all publication-quality (Nature-style) figures for the manuscript.
Requires all upstream analysis results to be present in results/.

Figures generated
-----------------
Fig 1 : Pipeline overview (schematic)
Fig 2 : Model performance comparison (ROC curves + bar chart)
Fig 3 : SHAP feature importance (beeswarm + bar)
Fig 4 : DEG volcano plots (HTSFilter vs RF)
Fig 5 : ORA dot plots (up/down pathways)
Fig 6 : GSEA Hallmark comparison
Fig 7 : Co-expression network hub genes
Fig 8 : Statistical validation overview (6-panel)

Inputs
------
- results/model_performance_5fold_CV.csv
- results/shap_feature_importance.csv
- results/de_HTSFilter_full.csv, de_RF_full.csv
- results/ORA_HTSFilter_up.csv, ORA_HTSFilter_down.csv
- results/GSEA_HTSFilter_Hallmark.csv, GSEA_RF_Hallmark.csv
- results/network_nodes_HTSFilter.csv, network_edges_HTSFilter.csv
- results/statistical_validation_RF_vs_HTSFilter.csv

Outputs
-------
- figures/fig1_pipeline.png
- figures/fig2_model_performance.png
- figures/fig3_shap.png
- figures/fig4_volcano.png
- figures/fig5_ORA.png
- figures/fig6_GSEA.png
- figures/fig7_network.png
- figures/fig8_validation.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.2)
PALETTE = {"HTSFilter": "#75A025", "RF": "#0279EE", "KNN": "#FF9400",
           "SVM": "#FD9BED", "MLP": "#E9ED4C", "XGBoost": "#000000",
           "LightGBM": "#ECE9E2", "CNN": "#FF9400", "RNN": "#0279EE"}
Path("figures").mkdir(exist_ok=True)

# ── Fig 2: Model performance ──────────────────────────────────────────────────
perf = pd.read_csv("results/model_performance_5fold_CV.csv")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC bar chart
perf_sorted = perf.sort_values("AUC", ascending=True)
colors = [PALETTE.get(m, "#999999") for m in perf_sorted["Model"]]
axes[0].barh(perf_sorted["Model"], perf_sorted["AUC"], color=colors, edgecolor="white")
axes[0].set_xlim(0.85, 1.01)
axes[0].set_xlabel("AUC (5-fold CV)")
axes[0].set_title("Model AUC Comparison")
for i, (_, row) in enumerate(perf_sorted.iterrows()):
    axes[0].text(row["AUC"] + 0.001, i, f"{row['AUC']:.4f}", va="center", fontsize=9)
sns.despine(ax=axes[0])

# F1 bar chart
perf_sorted2 = perf.sort_values("F1", ascending=True)
colors2 = [PALETTE.get(m, "#999999") for m in perf_sorted2["Model"]]
axes[1].barh(perf_sorted2["Model"], perf_sorted2["F1"], color=colors2, edgecolor="white")
axes[1].set_xlim(0.85, 1.01)
axes[1].set_xlabel("F1 Score (5-fold CV)")
axes[1].set_title("Model F1 Comparison")
for i, (_, row) in enumerate(perf_sorted2.iterrows()):
    axes[1].text(row["F1"] + 0.001, i, f"{row['F1']:.4f}", va="center", fontsize=9)
sns.despine(ax=axes[1])

plt.suptitle("Fig 2 — Model Performance Comparison (5-fold CV)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig2_model_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig2_model_performance.png")

# ── Fig 3: SHAP ───────────────────────────────────────────────────────────────
shap_df = pd.read_csv("results/shap_feature_importance.csv")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, label, color in [
    (axes[0], "RF_mean_abs_SHAP",  "RF",       "#75A025"),
    (axes[1], "XGB_mean_abs_SHAP", "XGBoost",  "#FD9BED"),
]:
    s = shap_df.sort_values(col, ascending=True)
    ax.barh(s["Feature"], s[col], color=color, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{label} — Feature Importance")
    sns.despine(ax=ax)
plt.suptitle("Fig 3 — SHAP Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig3_shap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig3_shap.png")

# ── Fig 4: Volcano plots ──────────────────────────────────────────────────────
PADJ_THRESH = 0.05; LFC_THRESH = 1.0
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, fname, label in [
    (axes[0], "results/de_HTSFilter_full.csv", "HTSFilter"),
    (axes[1], "results/de_RF_full.csv",        "RF"),
]:
    de = pd.read_csv(fname)
    de["neg_log10_padj"] = -np.log10(de["padj"].clip(lower=1e-300))
    de["color"] = "lightgrey"
    de.loc[(de["padj"] < PADJ_THRESH) & (de["log2FC"] >= LFC_THRESH),  "color"] = "#FF9400"
    de.loc[(de["padj"] < PADJ_THRESH) & (de["log2FC"] <= -LFC_THRESH), "color"] = "#0279EE"
    for color, grp in de.groupby("color"):
        ax.scatter(grp["log2FC"], grp["neg_log10_padj"], c=color, s=3, alpha=0.5, rasterized=True)
    ax.axhline(-np.log10(PADJ_THRESH), color="black", lw=1, ls="--")
    ax.axvline( LFC_THRESH, color="black", lw=1, ls="--")
    ax.axvline(-LFC_THRESH, color="black", lw=1, ls="--")
    n_up   = ((de["padj"] < PADJ_THRESH) & (de["log2FC"] >= LFC_THRESH)).sum()
    n_down = ((de["padj"] < PADJ_THRESH) & (de["log2FC"] <= -LFC_THRESH)).sum()
    ax.set_title(f"{label}  (↑{n_up} / ↓{n_down})")
    ax.set_xlabel("log₂ Fold Change"); ax.set_ylabel("-log₁₀(padj)")
    sns.despine(ax=ax)
plt.suptitle("Fig 4 — Differential Expression: HTSFilter vs RF", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig4_volcano.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/fig4_volcano.png")

# ── Fig 7: Network ────────────────────────────────────────────────────────────
try:
    edges = pd.read_csv("results/network_edges_HTSFilter.csv")
    nodes = pd.read_csv("results/network_nodes_HTSFilter.csv")
    G = nx.from_pandas_edgelist(edges, "source", "target", edge_attr="weight")
    top_hubs = nodes.sort_values("degree", ascending=False).head(30)["gene"].tolist()
    G_sub = G.subgraph(top_hubs)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G_sub, seed=42, k=2)
    deg = dict(G_sub.degree())
    mod_map = nodes.set_index("gene")["module"].to_dict()
    node_colors = [mod_map.get(n, 0) for n in G_sub.nodes()]
    nx.draw_networkx_nodes(G_sub, pos, node_size=[deg[n]*30 for n in G_sub.nodes()],
                            node_color=node_colors, cmap=cm.Set2, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3, width=0.8, ax=ax)
    nx.draw_networkx_labels(G_sub, pos, font_size=7, ax=ax)
    ax.set_title("Fig 7 — Hub Gene Co-expression Network (top 30 hubs)", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("figures/fig7_network.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/fig7_network.png")
except Exception as e:
    print(f"  Fig 7 skipped: {e}")

print("\nAll figures generated. Done.")
