#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end ML gene filtering pipeline for RNA-seq data.

Usage:
    python run_pipeline.py \
        --expr data/rnaseq_expression_BRCA.csv \
        --meta data/sample_info_BRCA.csv \
        --htsfilter data/htsfilter_labels.csv \
        --output results/ \
        --models LightGBM XGBoost RF MLP SVM KNN

Author: Soufiane El Atfa
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brca_ml_filtering.preprocessing import (
    load_expression_matrix, compute_gene_features, split_train_test
)
from brca_ml_filtering.models import train_all_models, predict_filtering, save_models
from brca_ml_filtering.evaluation import (
    evaluate_model, compute_filtering_agreement, compute_silhouette, bootstrap_auc_ci
)


def parse_args():
    parser = argparse.ArgumentParser(description="ML Gene Filtering Pipeline")
    parser.add_argument("--expr",       required=True, help="Expression matrix CSV (genes × samples)")
    parser.add_argument("--meta",       required=True, help="Sample metadata CSV")
    parser.add_argument("--htsfilter",  required=True, help="HTSFilter labels CSV (gene, label)")
    parser.add_argument("--output",     default="results/", help="Output directory")
    parser.add_argument("--models",     nargs="+",
                        default=["LightGBM", "XGBoost", "RF", "MLP", "SVM", "KNN"],
                        help="Models to train")
    parser.add_argument("--test-size",  type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed",       type=int,   default=42,  help="Random seed")
    parser.add_argument("--save-models", action="store_true", help="Save trained models to pickle")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("BRCA ML Gene Filtering Pipeline")
    print("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading expression matrix...")
    expr = load_expression_matrix(args.expr, log_transform=True, cpm_normalize=True)
    print(f"  Expression matrix: {expr.shape[0]} genes × {expr.shape[1]} samples")

    print("[1/5] Loading HTSFilter labels...")
    labels_df = pd.read_csv(args.htsfilter, index_col=0)
    labels = labels_df.iloc[:, 0].reindex(expr.index).fillna(0).astype(int)
    print(f"  Labels: {labels.sum()} kept, {(labels==0).sum()} filtered")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    print("\n[2/5] Computing gene features...")
    features = compute_gene_features(expr)
    print(f"  Features: {features.shape[1]} features × {features.shape[0]} genes")

    # ── Step 3: Train/test split ──────────────────────────────────────────────
    print("\n[3/5] Splitting train/test sets...")
    X_train, X_test, y_train, y_test = split_train_test(
        features, labels, test_size=args.test_size, random_state=args.seed
    )
    print(f"  Train: {len(X_train)} genes | Test: {len(X_test)} genes")

    # ── Step 4: Train models ──────────────────────────────────────────────────
    print(f"\n[4/5] Training {len(args.models)} models...")
    models = train_all_models(X_train, y_train, model_names=args.models, verbose=True)

    if args.save_models:
        save_models(models, os.path.join(args.output, "trained_models.pkl"))

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print("\n[5/5] Evaluating models...")
    results = []
    ref_kept = set(labels[labels == 1].index)
    all_genes = set(labels.index)

    for name, model in models.items():
        y_pred, y_proba = predict_filtering(model, X_test)
        metrics = evaluate_model(y_test.values, y_pred, y_proba)

        # Filtering agreement on full dataset
        full_pred, _ = predict_filtering(model, features)
        ml_kept = set(features.index[full_pred == 1])
        agreement = compute_filtering_agreement(ml_kept, ref_kept, all_genes)

        # Bootstrap CI
        ci = bootstrap_auc_ci(y_test.values, y_proba, n_bootstrap=500)

        row = {
            "Model":    name,
            "Test_AUC": metrics.get("auc", np.nan),
            "AUC_CI":   f"{ci['auc_lower']:.4f}–{ci['auc_upper']:.4f}",
            "Test_F1":  metrics["f1"],
            "Test_Acc": metrics["accuracy"],
            "Jaccard":  agreement["jaccard"],
            "Kappa":    agreement["kappa"],
            "TN": metrics["TN"], "FP": metrics["FP"],
            "FN": metrics["FN"], "TP": metrics["TP"],
        }
        results.append(row)
        print(f"  {name:12s}: AUC={row['Test_AUC']:.4f}, F1={row['Test_F1']:.4f}, "
              f"Jaccard={row['Jaccard']:.4f}")

    results_df = pd.DataFrame(results).sort_values("Test_AUC", ascending=False)
    out_path = os.path.join(args.output, "model_evaluation_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("\nTop model:", results_df.iloc[0]["Model"])
    print("=" * 60)


if __name__ == "__main__":
    main()
