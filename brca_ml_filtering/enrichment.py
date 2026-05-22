"""
enrichment.py
=============
Wrappers for differential expression (limma-voom via rpy2) and
pathway enrichment analysis (clusterProfiler via rpy2).

Note: Requires R with limma, edgeR, clusterProfiler, org.Hs.eg.db, msigdbr installed.
"""

import subprocess
import os
import pandas as pd
from typing import Optional


def run_deg_analysis(
    count_matrix_path: str,
    sample_info_path: str,
    output_dir: str,
    group_col: str = "disease_status",
    case_label: str = "Disease",
    control_label: str = "Normal",
    lfc_threshold: float = 1.0,
    padj_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Run limma-voom differential expression analysis via R subprocess.

    Parameters
    ----------
    count_matrix_path : str
        Path to raw count matrix (genes × samples CSV).
    sample_info_path : str
        Path to sample metadata CSV (must contain group_col column).
    output_dir : str
        Directory to save DEG results.
    group_col : str
        Column in sample_info for group labels.
    case_label : str
        Label for case group (e.g., "Disease").
    control_label : str
        Label for control group (e.g., "Normal").
    lfc_threshold : float
        Log2 fold-change threshold for significance.
    padj_threshold : float
        Adjusted p-value threshold for significance.

    Returns
    -------
    pd.DataFrame
        Significant DEGs with columns: gene, logFC, AveExpr, t, P.Value, adj.P.Val, B
    """
    os.makedirs(output_dir, exist_ok=True)

    r_script = f"""
suppressPackageStartupMessages({{
  library(limma); library(edgeR)
}})

counts <- read.csv("{count_matrix_path}", row.names=1, check.names=FALSE)
meta   <- read.csv("{sample_info_path}")
meta   <- meta[meta${group_col} %in% c("{case_label}", "{control_label}"), ]
counts <- counts[, colnames(counts) %in% meta$sample_id]
meta   <- meta[match(colnames(counts), meta$sample_id), ]

group <- factor(meta${group_col}, levels=c("{control_label}", "{case_label}"))
dge   <- DGEList(counts=counts, group=group)
dge   <- filterByExpr(dge)
dge   <- calcNormFactors(dge)

design <- model.matrix(~group)
v      <- voom(dge, design, plot=FALSE)
fit    <- lmFit(v, design)
fit    <- eBayes(fit)

res_full <- topTable(fit, coef=2, number=Inf, sort.by="P")
res_full$gene <- rownames(res_full)
res_sig  <- res_full[abs(res_full$logFC) > {lfc_threshold} & res_full$adj.P.Val < {padj_threshold}, ]

write.csv(res_full, "{output_dir}/limma_results_full.csv", row.names=FALSE)
write.csv(res_sig,  "{output_dir}/limma_results_sig.csv",  row.names=FALSE)
cat(sprintf("DEGs: %d (up=%d, down=%d)\\n",
    nrow(res_sig), sum(res_sig$logFC > 0), sum(res_sig$logFC < 0)))
"""

    script_path = os.path.join(output_dir, "_run_limma.R")
    with open(script_path, "w") as f:
        f.write(r_script)

    result = subprocess.run(["Rscript", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"limma-voom failed:\n{result.stderr}")

    print(result.stdout)
    return pd.read_csv(os.path.join(output_dir, "limma_results_sig.csv"))


def run_pathway_enrichment(
    deg_full_path: str,
    deg_sig_path: str,
    output_dir: str,
    organism: str = "hsa",
    padj_threshold: float = 0.05,
) -> dict:
    """
    Run KEGG ORA, GO-BP ORA, and Hallmark GSEA via R subprocess.

    Parameters
    ----------
    deg_full_path : str
        Path to full DEG results CSV (for GSEA ranked list).
    deg_sig_path : str
        Path to significant DEG results CSV (for ORA).
    output_dir : str
        Directory to save enrichment results.
    organism : str
        KEGG organism code (default: "hsa" for human).
    padj_threshold : float
        Adjusted p-value threshold.

    Returns
    -------
    dict
        {kegg_ora, go_bp_ora, gsea_hallmark, gsea_kegg} DataFrames
    """
    os.makedirs(output_dir, exist_ok=True)

    r_script = f"""
suppressPackageStartupMessages({{
  library(clusterProfiler); library(org.Hs.eg.db); library(msigdbr); library(dplyr)
}})

res_full <- read.csv("{deg_full_path}")
res_sig  <- read.csv("{deg_sig_path}")

map_entrez <- function(symbols) {{
  bitr(symbols, fromType="SYMBOL", toType="ENTREZID", OrgDb=org.Hs.eg.db)$ENTREZID
}}

entrez_all  <- map_entrez(res_full$gene)
entrez_up   <- map_entrez(res_sig$gene[res_sig$logFC > 0])
entrez_down <- map_entrez(res_sig$gene[res_sig$logFC < 0])

# KEGG ORA
kegg_up   <- enrichKEGG(gene=entrez_up,   organism="{organism}", universe=entrez_all,
                         pvalueCutoff={padj_threshold}, pAdjustMethod="BH")
kegg_down <- enrichKEGG(gene=entrez_down, organism="{organism}", universe=entrez_all,
                         pvalueCutoff={padj_threshold}, pAdjustMethod="BH")
ku <- as.data.frame(kegg_up);  ku$direction <- "Up"
kd <- as.data.frame(kegg_down); kd$direction <- "Down"
write.csv(rbind(ku, kd), "{output_dir}/kegg_ora.csv", row.names=FALSE)

# GO-BP ORA
go_up   <- enrichGO(gene=entrez_up,   OrgDb=org.Hs.eg.db, ont="BP",
                     universe=entrez_all, pvalueCutoff={padj_threshold},
                     pAdjustMethod="BH", readable=TRUE)
go_down <- enrichGO(gene=entrez_down, OrgDb=org.Hs.eg.db, ont="BP",
                     universe=entrez_all, pvalueCutoff={padj_threshold},
                     pAdjustMethod="BH", readable=TRUE)
gu <- as.data.frame(go_up);  gu$direction <- "Up"
gd <- as.data.frame(go_down); gd$direction <- "Down"
write.csv(rbind(gu, gd), "{output_dir}/go_bp_ora.csv", row.names=FALSE)

# Hallmark GSEA
hallmark_sets <- msigdbr(species="Homo sapiens", collection="H") %>%
  dplyr::select(gs_name, entrez_gene) %>%
  dplyr::mutate(entrez_gene = as.character(entrez_gene))
entrez_map <- bitr(res_full$gene, fromType="SYMBOL", toType="ENTREZID", OrgDb=org.Hs.eg.db)
res_ranked <- merge(res_full, entrez_map, by.x="gene", by.y="SYMBOL")
ranked_list <- setNames(res_ranked$logFC, res_ranked$ENTREZID)
ranked_list <- sort(ranked_list[!duplicated(names(ranked_list))], decreasing=TRUE)
gsea_h <- GSEA(geneList=ranked_list, TERM2GENE=hallmark_sets,
               pvalueCutoff={padj_threshold}, pAdjustMethod="BH",
               minGSSize=15, maxGSSize=500, eps=0, seed=42)
write.csv(as.data.frame(gsea_h), "{output_dir}/gsea_hallmark.csv", row.names=FALSE)

cat("Enrichment complete.\\n")
"""

    script_path = os.path.join(output_dir, "_run_enrichment.R")
    with open(script_path, "w") as f:
        f.write(r_script)

    result = subprocess.run(["Rscript", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Enrichment failed:\n{result.stderr}")

    print(result.stdout)
    return {
        "kegg_ora":      pd.read_csv(os.path.join(output_dir, "kegg_ora.csv")),
        "go_bp_ora":     pd.read_csv(os.path.join(output_dir, "go_bp_ora.csv")),
        "gsea_hallmark": pd.read_csv(os.path.join(output_dir, "gsea_hallmark.csv")),
    }
