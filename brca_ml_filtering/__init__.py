"""
BRCA ML Filtering Pipeline
==========================
Machine Learning-based gene filtering for RNA-seq data.
Benchmarked against HTSFilter on TCGA BRCA data.

Authors: Soufiane El Atfa
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Soufiane El Atfa"

from .preprocessing import load_expression_matrix, compute_gene_features, split_train_test
from .models import train_all_models, predict_filtering
from .evaluation import evaluate_model, compute_jaccard, compute_silhouette
from .enrichment import run_deg_analysis, run_pathway_enrichment

__all__ = [
    "load_expression_matrix",
    "compute_gene_features",
    "split_train_test",
    "train_all_models",
    "predict_filtering",
    "evaluate_model",
    "compute_jaccard",
    "compute_silhouette",
    "run_deg_analysis",
    "run_pathway_enrichment",
]
