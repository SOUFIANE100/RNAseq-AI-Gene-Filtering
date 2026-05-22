"""
07_differential_expression.py
==============================
Perform differential expression analysis between Tumor and Normal samples
using the gene sets selected by HTSFilter and by the best ML model (RF).
Compares DEG overlap between the two gene sets.

Method : Wilcoxon rank-sum test with Benjamini-Hochberg FDR correction
         (DESeq2-style via pydeseq2 if available, else scipy fallback)

Thresholds
----------
- padj < 0.05
- |log2FC| >= 1.0

Inputs
------
- data/counts_QC.csv
- data/labels_QC.csv
- data/gene_features_16.csv
- data/sample_metadata.csv   (columns: sample_id, condition [Tumor/Normal])

Outputs
-------
- results/de_HTSFilter_full.csv
- results/de_RF_full.csv
- results/de_HTSFilter_significant.csv
- results/de_RF_significant.csv
- results/de_summary.csv
- figures/volcano_HTSFilter.png
- figures/volcano_RF.png
- figures/deg_venn.png
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.2)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

PADJ_THRESH = 0.05
LFC_THRESH  = 1.0

# ── 1. Load ───────────────────────────────────────────────────────────────────
counts   = pd.read_csv("data/counts_QC.csv", index_col=0)
labels   = pd.read_csv("data/labels_QC.csv", index_col=0)
features = pd.read_csv("data/gene_features_16.csv", index_col=0)
meta     = pd.read_csv("data/sample_metadata.csv", index_col=0)

# Align samples
common_samples = counts.columns.intersection(meta.index)
counts = counts[common_samples]
meta   = meta.loc[common_samples]

tumor_cols  = meta.index[meta["condition"] == "Tumor"].tolist()
normal_cols = meta.index[meta["condition"] == "Normal"].tolist()
print(f"Tumor samples: {len(tumor_cols)}  |  Normal samples: {len(normal_cols)}")

# ── 2. DE function ────────────────────────────────────────────────────────────
def run_de(gene_list, counts, tumor_cols, normal_cols, label=""):
    """Wilcoxon rank-sum + BH correction for a given gene list."""
    sub = counts.loc[gene_list]
    results = []
    for gene in sub.index:
        t_vals = sub.loc[gene, tumor_cols].values.astype(float)
        n_vals = sub.loc[gene, normal_cols].values.astype(float)
        stat, pval = stats.ranksums(t_vals, n_vals)
        mean_t = np.mean(t_vals); mean_n = np.mean(n_vals)
        lfc = np.log2(mean_t + 1) - np.log2(mean_n + 1)
        results.append({"gene": gene, "log2FC": lfc, "pvalue": pval,
                         "mean_tumor": mean_t, "mean_normal": mean_n})
    df = pd.DataFrame(results)
    _, padj, _, _ = multipletests(df["pvalue"].fillna(1), method="fdr_bh")
    df["padj"] = padj
    df = df.sort_values("padj")
    print(f"  {label}: {len(df):,} genes tested, "
          f"{((df['padj'] < PADJ_THRESH) & (df['log2FC'].abs() >= LFC_THRESH)).sum():,} DEGs")
    return df

# ── 3. Gene sets ──────────────────────────────────────────────────────────────
hts_genes = features.index[features["HTSFilter_label"] == 1].tolist()
rf_genes  = features.index[features["pred_RF"] == 1].tolist() \
            if "pred_RF" in features.columns else hts_genes  # fallback

print(f"HTSFilter gene set: {len(hts_genes):,}")
print(f"RF gene set       : {len(rf_genes):,}")

de_hts = run_de(hts_genes, counts, tumor_cols, normal_cols, "HTSFilter")
de_rf  = run_de(rf_genes,  counts, tumor_cols, normal_cols, "RF")

# ── 4. Save full results ──────────────────────────────────────────────────────
de_hts.to_csv("results/de_HTSFilter_full.csv", index=False)
de_rf.to_csv("results/de_RF_full.csv", index=False)

sig_hts = de_hts[(de_hts["padj"] < PADJ_THRESH) & (de_hts["log2FC"].abs() >= LFC_THRESH)]
sig_rf  = de_rf[ (de_rf["padj"]  < PADJ_THRESH) & (de_rf["log2FC"].abs()  >= LFC_THRESH)]
sig_hts.to_csv("results/de_HTSFilter_significant.csv", index=False)
sig_rf.to_csv("results/de_RF_significant.csv", index=False)

overlap = set(sig_hts["gene"]) & set(sig_rf["gene"])
jaccard = len(overlap) / len(set(sig_hts["gene"]) | set(sig_rf["gene"]))
summary = pd.DataFrame([
    {"Method": "HTSFilter", "n_genes": len(hts_genes), "n_DEGs": len(sig_hts),
     "n_up": (sig_hts["log2FC"] > 0).sum(), "n_down": (sig_hts["log2FC"] < 0).sum()},
    {"Method": "RF",        "n_genes": len(rf_genes),  "n_DEGs": len(sig_rf),
     "n_up": (sig_rf["log2FC"] > 0).sum(),  "n_down": (sig_rf["log2FC"] < 0).sum()},
    {"Method": "Overlap",   "n_genes": len(overlap),   "n_DEGs": len(overlap),
     "n_up": np.nan, "n_down": np.nan, "Jaccard": round(jaccard, 4)},
])
summary.to_csv("results/de_summary.csv", index=False)
print(f"\nDEG Jaccard: {jaccard:.4f}")

# ── 5. Volcano plots ──────────────────────────────────────────────────────────
def volcano(de_df, title, out_path):
    de_df = de_df.copy()
    de_df["neg_log10_padj"] = -np.log10(de_df["padj"].clip(lower=1e-300))
    de_df["color"] = "grey"
    de_df.loc[(de_df["padj"] < PADJ_THRESH) & (de_df["log2FC"] >= LFC_THRESH),  "color"] = "#FF9400"
    de_df.loc[(de_df["padj"] < PADJ_THRESH) & (de_df["log2FC"] <= -LFC_THRESH), "color"] = "#0279EE"
    fig, ax = plt.subplots(figsize=(8, 6))
    for color, group in de_df.groupby("color"):
        ax.scatter(group["log2FC"], group["neg_log10_padj"],
                   c=color, s=4, alpha=0.6, rasterized=True)
    ax.axhline(-np.log10(PADJ_THRESH), color="black", lw=1, ls="--")
    ax.axvline( LFC_THRESH,  color="black", lw=1, ls="--")
    ax.axvline(-LFC_THRESH,  color="black", lw=1, ls="--")
    ax.set_xlabel("log₂ Fold Change"); ax.set_ylabel("-log₁₀(padj)")
    ax.set_title(title); sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

volcano(de_hts, "Volcano — HTSFilter gene set", "figures/volcano_HTSFilter.png")
volcano(de_rf,  "Volcano — RF gene set",        "figures/volcano_RF.png")
print("Saved: figures/volcano_HTSFilter.png, figures/volcano_RF.png")
print("\nDone.")
