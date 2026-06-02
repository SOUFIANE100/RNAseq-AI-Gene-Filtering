"""
08_enrichment_analysis.py
==========================
Perform Over-Representation Analysis (ORA) and Gene Set Enrichment Analysis
(GSEA) on DEGs from HTSFilter and RF gene sets.

ORA  : gseapy.enrichr  (KEGG_2021_Human, Reactome_2022, GO_Biological_Process_2021)
GSEA : gseapy.prerank  (MSigDB Hallmark, KEGG_2021_Human)

Thresholds
----------
ORA  : padj < 0.05
GSEA : padj < 0.05, |NES| >= 1.0



"""

import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.1)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

PADJ_ORA  = 0.05
PADJ_GSEA = 0.05
NES_THRESH = 1.0
LFC_THRESH = 1.0

# ── 1. Load DE results ────────────────────────────────────────────────────────
de_hts = pd.read_csv("results/de_HTSFilter_full.csv")
de_rf  = pd.read_csv("results/de_RF_full.csv")

def get_gene_lists(de_df, padj_thresh=0.05, lfc_thresh=1.0):
    sig = de_df[(de_df["padj"] < padj_thresh) & (de_df["log2FC"].abs() >= lfc_thresh)]
    up   = sig[sig["log2FC"] > 0]["gene"].tolist()
    down = sig[sig["log2FC"] < 0]["gene"].tolist()
    return up, down

hts_up, hts_down = get_gene_lists(de_hts)
rf_up,  rf_down  = get_gene_lists(de_rf)
print(f"HTSFilter: {len(hts_up)} up, {len(hts_down)} down")
print(f"RF       : {len(rf_up)} up, {len(rf_down)} down")

# ── 2. ORA ────────────────────────────────────────────────────────────────────
gene_sets_ora = ["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2021"]

def run_ora(gene_list, label, direction):
    if len(gene_list) < 5:
        print(f"  Skipping ORA {label} {direction}: too few genes ({len(gene_list)})")
        return pd.DataFrame()
    try:
        enr = gp.enrichr(gene_list=gene_list, gene_sets=gene_sets_ora,
                          organism="Human", outdir=None, verbose=False)
        df = enr.results
        df = df[df["Adjusted P-value"] < PADJ_ORA].copy()
        df["Method"] = label; df["Direction"] = direction
        return df
    except Exception as e:
        print(f"  ORA error {label} {direction}: {e}")
        return pd.DataFrame()

ora_hts_up   = run_ora(hts_up,   "HTSFilter", "up")
ora_hts_down = run_ora(hts_down, "HTSFilter", "down")
ora_rf_up    = run_ora(rf_up,    "RF",        "up")
ora_rf_down  = run_ora(rf_down,  "RF",        "down")

for df, fname in [(ora_hts_up,   "results/ORA_HTSFilter_up.csv"),
                   (ora_hts_down, "results/ORA_HTSFilter_down.csv"),
                   (ora_rf_up,    "results/ORA_RF_up.csv"),
                   (ora_rf_down,  "results/ORA_RF_down.csv")]:
    df.to_csv(fname, index=False)
    print(f"  Saved: {fname}  ({len(df)} pathways)")

# ── 3. GSEA ───────────────────────────────────────────────────────────────────
def make_ranking(de_df):
    de_df = de_df.copy()
    de_df["rank"] = np.sign(de_df["log2FC"]) * (-np.log10(de_df["pvalue"].clip(lower=1e-300)))
    return de_df.set_index("gene")["rank"].sort_values(ascending=False)

rnk_hts = make_ranking(de_hts)
rnk_rf  = make_ranking(de_rf)

def run_gsea(ranking, gene_set_name, label):
    try:
        res = gp.prerank(rnk=ranking, gene_sets=gene_set_name,
                          min_size=15, max_size=500, permutation_num=1000,
                          outdir=None, verbose=False, seed=42)
        df = res.res2d
        df = df[df["FDR q-val"] < PADJ_GSEA].copy()
        df["Method"] = label
        return df
    except Exception as e:
        print(f"  GSEA error {label} {gene_set_name}: {e}")
        return pd.DataFrame()

print("\nRunning GSEA …")
gsea_hts_hall = run_gsea(rnk_hts, "MSigDB_Hallmark_2020", "HTSFilter")
gsea_hts_kegg = run_gsea(rnk_hts, "KEGG_2021_Human",      "HTSFilter")
gsea_rf_hall  = run_gsea(rnk_rf,  "MSigDB_Hallmark_2020", "RF")
gsea_rf_kegg  = run_gsea(rnk_rf,  "KEGG_2021_Human",      "RF")

for df, fname in [(gsea_hts_hall, "results/GSEA_HTSFilter_Hallmark.csv"),
                   (gsea_hts_kegg, "results/GSEA_HTSFilter_KEGG.csv"),
                   (gsea_rf_hall,  "results/GSEA_RF_Hallmark.csv"),
                   (gsea_rf_kegg,  "results/GSEA_RF_KEGG.csv")]:
    df.to_csv(fname, index=False)
    print(f"  Saved: {fname}  ({len(df)} pathways)")

# ── 4. Pathway comparison NES table ──────────────────────────────────────────
def merge_nes(df_hts, df_rf, gene_set):
    if df_hts.empty or df_rf.empty:
        return pd.DataFrame()
    col = "NES" if "NES" in df_hts.columns else "nes"
    name_col = "Term" if "Term" in df_hts.columns else "term"
    m = df_hts[[name_col, col]].rename(columns={col: "NES_HTSFilter", name_col: "Pathway"})
    r = df_rf[[name_col, col]].rename(columns={col: "NES_RF", name_col: "Pathway"})
    merged = m.merge(r, on="Pathway", how="outer")
    merged["GeneSet"] = gene_set
    return merged

nes_hall = merge_nes(gsea_hts_hall, gsea_rf_hall, "Hallmark")
nes_kegg = merge_nes(gsea_hts_kegg, gsea_rf_kegg, "KEGG")
nes_all  = pd.concat([nes_hall, nes_kegg], ignore_index=True)
nes_all.to_csv("results/GSEA_pathway_comparison_NES.csv", index=False)
print(f"  Saved: results/GSEA_pathway_comparison_NES.csv  ({len(nes_all)} pathways)")

# ── 5. Figures ────────────────────────────────────────────────────────────────
# ORA dot plot (top 15 pathways, HTSFilter up)
if not ora_hts_up.empty:
    top = ora_hts_up.nsmallest(15, "Adjusted P-value").copy()
    top["neg_log10_padj"] = -np.log10(top["Adjusted P-value"])
    top["Overlap_ratio"] = top["Overlap"].apply(
        lambda x: int(x.split("/")[0]) / int(x.split("/")[1]) if "/" in str(x) else 0)
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(top["neg_log10_padj"], top["Term"],
                    s=top["Overlap_ratio"] * 500, c=top["neg_log10_padj"],
                    cmap="YlOrRd", edgecolors="grey", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="-log₁₀(padj)")
    ax.set_xlabel("-log₁₀(Adjusted P-value)")
    ax.set_title("ORA — HTSFilter Up-regulated Pathways")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig("figures/ORA_dotplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/ORA_dotplot.png")

# GSEA Hallmark comparison bar chart
if not nes_hall.empty:
    nes_hall_sorted = nes_hall.dropna(subset=["NES_HTSFilter"]).sort_values("NES_HTSFilter")
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(nes_hall_sorted))
    ax.barh(y_pos, nes_hall_sorted["NES_HTSFilter"], color="#75A025", alpha=0.7, label="HTSFilter")
    if "NES_RF" in nes_hall_sorted.columns:
        ax.barh(y_pos, nes_hall_sorted["NES_RF"], color="#0279EE", alpha=0.5, label="RF")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(nes_hall_sorted["Pathway"].str.replace("HALLMARK_",""), fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("NES"); ax.set_title("GSEA Hallmark — HTSFilter vs RF")
    ax.legend(); sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig("figures/GSEA_Hallmark_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/GSEA_Hallmark_comparison.png")

print("\nDone.")
