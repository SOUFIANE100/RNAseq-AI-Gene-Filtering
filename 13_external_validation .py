#!/usr/bin/env python3
"""
11_external_validation_REAL.py
===============================
Validation externe zero-shot du surrogate LightGBM entraîné sur TCGA-BRCA,
appliqué sans ré-entraînement à TCGA-LUAD et TCGA-COAD.

"""

import argparse
import os
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, jaccard_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

sns.set_theme(style="ticks", font_scale=1.1)

# ── Phylo color palette ───────────────────────────────────────────────────────
COLORS = {
    "BRCA":     "#000000",
    "LUAD":     "#0279EE",
    "COAD":     "#75A025",
}

# ── Cancer types (recount2 / TCGA labels) ────────────────────────────────────
CANCER_MAP = {
    "BRCA": "Breast Invasive Carcinoma",
    "LUAD": "Lung Adenocarcinoma",
    "COAD": "Colon Adenocarcinoma",
}

# ── LightGBM hyperparameters (article §2.3) ───────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=7,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — 18 features (article §2.2, matching feature_matrix_all.csv)
# ─────────────────────────────────────────────────────────────────────────────

def compute_18_features(count_matrix):
    """
    Compute 18 gene-level statistical features from a raw count matrix.

    Column names match feature_matrix_all.csv and article figures exactly.

    Parameters
    ----------
    count_matrix : np.ndarray, shape (n_samples, n_genes), dtype int32
        Raw RNA-seq count matrix (samples × genes).

    Returns
    -------
    feat_matrix : np.ndarray, shape (n_genes, 18), dtype float32
    feat_names  : list of str, length 18
    """
    X = count_matrix.astype(np.float64)   # (n_samples, n_genes)

    # ── Basic statistics ──────────────────────────────────────────────────
    mean_expr   = X.mean(axis=0)
    var_expr    = X.var(axis=0)
    std_expr    = X.std(axis=0)
    median_expr = np.median(X, axis=0)

    # Coefficient of variation (avoid division by zero)
    cv_expr     = np.where(mean_expr > 0, std_expr / mean_expr, 0.0)

    # ── Sparsity & percentiles ────────────────────────────────────────────
    sparsity    = (X == 0).mean(axis=0)
    q25         = np.percentile(X, 25, axis=0)
    q75         = np.percentile(X, 75, axis=0)
    iqr_expr    = q75 - q25

    # ── Extremes ──────────────────────────────────────────────────────────
    min_expr    = X.min(axis=0)
    max_expr    = X.max(axis=0)
    range_expr  = max_expr - min_expr

    # ── Shape statistics ──────────────────────────────────────────────────
    skewness    = stats.skew(X, axis=0)
    kurtosis    = stats.kurtosis(X, axis=0)   # excess kurtosis

    # ── Non-zero mean ─────────────────────────────────────────────────────
    nonzero_sum   = X.sum(axis=0)
    nonzero_count = (X > 0).sum(axis=0).astype(float)
    mean_nonzero  = np.where(nonzero_count > 0,
                             nonzero_sum / nonzero_count, 0.0)

    # ── Log-transformed statistics ────────────────────────────────────────
    log_mean    = np.log1p(mean_expr)
    log_var     = np.log1p(var_expr)

    # ── Percentage above mean ─────────────────────────────────────────────
    pct_above_mean = (X > mean_expr).mean(axis=0)

    # ── Assemble in article order ─────────────────────────────────────────
    feat_names = [
        'mean_expr', 'var_expr', 'std_expr', 'median_expr', 'cv_expr',
        'sparsity',  'q25',      'q75',      'iqr_expr',
        'min_expr',  'max_expr', 'range_expr',
        'skewness',  'kurtosis',
        'mean_nonzero',
        'log_mean',  'log_var',
        'pct_above_mean',
    ]

    feat_list = [
        mean_expr, var_expr, std_expr, median_expr, cv_expr,
        sparsity,  q25,      q75,      iqr_expr,
        min_expr,  max_expr, range_expr,
        skewness,  kurtosis,
        mean_nonzero,
        log_mean,  log_var,
        pct_above_mean,
    ]

    feat_matrix = np.column_stack(feat_list).astype(np.float32)
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    assert feat_matrix.shape[1] == 18, \
        f"Expected 18 features, got {feat_matrix.shape[1]}"

    return feat_matrix, feat_names


# ─────────────────────────────────────────────────────────────────────────────
# HTSFilter EQUIVALENT (CPM-based Jaccard maximization)
# ─────────────────────────────────────────────────────────────────────────────

def htsfilter_cpm(counts, n_thresholds=100, s_min=0.1, s_max=10.0,
                  n_pairs=300, seed=42):
    """
    HTSFilter-equivalent on CPM-normalized counts.

    Maximizes global Jaccard similarity across sample pairs over a grid of
    CPM thresholds. Gene retained if max_CPM > s*.

    Parameters
    ----------
    counts       : np.ndarray, shape (n_samples, n_genes), raw counts
    n_thresholds : int, number of threshold values to scan (default 100)
    s_min        : float, minimum CPM threshold (default 0.1)
    s_max        : float, maximum CPM threshold (default 10.0)
    n_pairs      : int, max number of sample pairs to evaluate (default 300)
    seed         : int, random seed for pair subsampling (default 42)

    Returns
    -------
    labels         : np.ndarray of int (0/1), shape (n_genes,)
    s_star         : float, optimal CPM threshold
    global_jaccard : np.ndarray, shape (n_thresholds,)
    thresholds     : np.ndarray, shape (n_thresholds,)
    max_cpm        : np.ndarray, shape (n_genes,)
    """
    lib_sizes = counts.sum(axis=1, keepdims=True)
    lib_sizes = np.where(lib_sizes == 0, 1, lib_sizes)
    norm = counts / lib_sizes * 1e6   # CPM

    thresholds     = np.linspace(s_min, s_max, n_thresholds)
    global_jaccard = np.zeros(n_thresholds)
    n_samples      = counts.shape[0]

    all_pairs = [(i, j) for i in range(n_samples)
                         for j in range(i + 1, n_samples)]
    rng = np.random.default_rng(seed)
    if len(all_pairs) > n_pairs:
        sel   = rng.choice(len(all_pairs), n_pairs, replace=False)
        pairs = [all_pairs[k] for k in sel]
    else:
        pairs = all_pairs

    for t_idx, s in enumerate(thresholds):
        bin_matrix = norm > s
        jvals = []
        for i, j in pairs:
            a     = np.sum(bin_matrix[i] & bin_matrix[j])
            b     = np.sum(bin_matrix[i] & ~bin_matrix[j])
            c     = np.sum(~bin_matrix[i] & bin_matrix[j])
            denom = a + b + c
            jvals.append(a / denom if denom > 0 else 0.0)
        global_jaccard[t_idx] = np.mean(jvals)

    s_star_idx = np.argmax(global_jaccard)
    s_star     = thresholds[s_star_idx]
    max_cpm    = norm.max(axis=0)
    labels     = (max_cpm > s_star).astype(int)

    return labels, s_star, global_jaccard, thresholds, max_cpm


# ─────────────────────────────────────────────────────────────────────────────
# LOAD HDF5 AND PROCESS EACH CANCER TYPE
# ─────────────────────────────────────────────────────────────────────────────

def load_and_process(h5_path):
    """
    Load tcga_matrix.h5 and compute 18 features + HTSFilter labels
    for BRCA, LUAD, and COAD.

    Parameters
    ----------
    h5_path : str, path to tcga_matrix.h5

    Returns
    -------
    results : dict keyed by cancer abbreviation ('BRCA', 'LUAD', 'COAD')
    """
    import h5py

    log.info("Loading HDF5: %s", h5_path)
    f = h5py.File(h5_path, "r")

    cancer_types = np.array([c.decode() for c in f["meta"]["cancertype"][:]])
    sample_type  = np.array([s.decode() for s in
                             f["meta"]["gdc_cases.samples.sample_type"][:]])
    gene_names   = np.array([g.decode() for g in f["meta"]["genes"][:]])
    expr_full    = f["data"]["expression"]   # (11284, 25150) int32

    log.info("Matrix shape: %s, dtype: %s", expr_full.shape, expr_full.dtype)

    results = {}

    for cancer_abbr, cancer_full in CANCER_MAP.items():
        log.info("Processing %s (%s)", cancer_abbr, cancer_full)

        ct_idx  = np.where(cancer_types == cancer_full)[0]
        st_arr  = sample_type[ct_idx]
        keep_mask       = np.isin(st_arr, ["Primary Tumor", "Solid Tissue Normal"])
        ct_idx_filtered = ct_idx[keep_mask]

        n_tumor  = int(np.sum(st_arr == "Primary Tumor"))
        n_normal = int(np.sum(st_arr == "Solid Tissue Normal"))
        log.info("  Samples: %d (%d tumor, %d normal)",
                 len(ct_idx_filtered), n_tumor, n_normal)

        counts = expr_full[ct_idx_filtered, :]
        log.info("  Count matrix: %s, dtype=%s", counts.shape, counts.dtype)

        log.info("  Computing 18 features (article names)...")
        feat_matrix, feat_names = compute_18_features(counts)
        log.info("  Feature matrix: %s | names: %s", feat_matrix.shape, feat_names)

        log.info("  Running HTSFilter-CPM (100 thresholds, 0.1–10 CPM)...")
        labels, s_star, gj, thr, max_cpm = htsfilter_cpm(
            counts.astype(np.float64))
        n_retained = int(labels.sum())
        n_filtered = int((labels == 0).sum())
        log.info("  s* = %.3f CPM | retained=%d (%.1f%%) | filtered=%d",
                 s_star, n_retained, 100 * n_retained / len(labels), n_filtered)

        results[cancer_abbr] = {
            "counts":         counts,
            "feat_matrix":    feat_matrix,
            "feat_names":     feat_names,
            "labels":         labels,
            "s_star":         s_star,
            "global_jaccard": gj,
            "thresholds":     thr,
            "max_cpm":        max_cpm,
            "n_samples":      len(ct_idx_filtered),
            "n_genes":        counts.shape[1],
            "n_retained":     n_retained,
            "n_filtered":     n_filtered,
            "gene_names":     gene_names,
        }

    f.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN LIGHTGBM ON BRCA, TRANSFER TO LUAD AND COAD
# ─────────────────────────────────────────────────────────────────────────────

def run_transfer_evaluation(results):
    """
    Train LightGBM on BRCA (80/20 split), apply frozen model to LUAD and COAD.

    Returns
    -------
    transfer_results : dict keyed by cancer abbreviation
    scaler           : fitted StandardScaler (BRCA)
    lgb_model        : fitted LightGBM model (BRCA)
    X_train, y_train : BRCA training data
    """
    log.info("Training LightGBM surrogate on BRCA (80/20 split)...")

    X_brca = results["BRCA"]["feat_matrix"]   # (25150, 18)
    y_brca = results["BRCA"]["labels"]        # (25150,)

    # StandardScaler fitted on BRCA only (no data leakage)
    scaler        = StandardScaler()
    X_brca_scaled = scaler.fit_transform(X_brca)

    X_train, X_test, y_train, y_test = train_test_split(
        X_brca_scaled, y_brca,
        test_size=0.2, random_state=42, stratify=y_brca)

    log.info("  Train: %s | Test: %s", X_train.shape, X_test.shape)

    lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    lgb_model.fit(X_train, y_train)

    # BRCA test set
    y_prob_brca = lgb_model.predict_proba(X_test)[:, 1]
    y_pred_brca = lgb_model.predict(X_test)
    auc_brca    = roc_auc_score(y_test, y_prob_brca)
    f1_brca     = f1_score(y_test, y_pred_brca)
    jac_brca    = jaccard_score(y_test, y_pred_brca)
    acc_brca    = accuracy_score(y_test, y_pred_brca)
    disc_brca   = int(np.sum(y_pred_brca != y_test))

    log.info("  BRCA (test set): AUC=%.5f, F1=%.4f, Jaccard=%.4f, Acc=%.4f",
             auc_brca, f1_brca, jac_brca, acc_brca)

    transfer_results = {
        "BRCA": {
            "auc": auc_brca, "f1": f1_brca, "jaccard": jac_brca, "acc": acc_brca,
            "n_retained_hts": int(y_test.sum()),
            "n_retained_ml":  int(y_pred_brca.sum()),
            "discordant":     disc_brca,
            "y_pred": y_pred_brca, "y_prob": y_prob_brca, "y_true": y_test,
        }
    }

    # Decision Tree baseline (trained on BRCA, applied to transfer cohorts)
    dt = DecisionTreeClassifier(
        max_depth=5, min_samples_split=10,
        min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)

    for cancer_abbr in ["LUAD", "COAD"]:
        log.info("Transfer evaluation on %s (zero-shot)...", cancer_abbr)

        X_ext      = results[cancer_abbr]["feat_matrix"]
        y_ext      = results[cancer_abbr]["labels"]
        feat_names = results[cancer_abbr]["feat_names"]

        # Scale using BRCA scaler — no retraining
        X_ext_scaled = scaler.transform(X_ext)

        y_prob = lgb_model.predict_proba(X_ext_scaled)[:, 1]
        y_pred = lgb_model.predict(X_ext_scaled)

        auc  = roc_auc_score(y_ext, y_prob)
        f1   = f1_score(y_ext, y_pred)
        jac  = jaccard_score(y_ext, y_pred)
        acc  = accuracy_score(y_ext, y_pred)
        disc = int(np.sum(y_pred != y_ext))

        log.info("  AUC=%.5f, F1=%.4f, Jaccard=%.4f, Acc=%.4f", auc, f1, jac, acc)
        log.info("  HTSFilter=%d, LightGBM=%d, Discordant=%d",
                 int(y_ext.sum()), int(y_pred.sum()), disc)

        # Baselines (IQR, Sparsity, Decision Tree)
        iqr_idx  = feat_names.index("iqr_expr")
        spar_idx = feat_names.index("sparsity")
        iqr_vals  = X_ext[:, iqr_idx]
        spar_vals = X_ext[:, spar_idx]

        y_iqr  = (iqr_vals  > np.median(iqr_vals)).astype(int)
        y_spar = (spar_vals < 0.5).astype(int)
        y_dt   = dt.predict(X_ext_scaled)

        jac_iqr  = jaccard_score(y_ext, y_iqr)
        jac_spar = jaccard_score(y_ext, y_spar)
        jac_dt   = jaccard_score(y_ext, y_dt)
        auc_iqr  = roc_auc_score(y_ext, iqr_vals)
        auc_spar = roc_auc_score(y_ext, 1 - spar_vals)
        auc_dt   = roc_auc_score(y_ext, dt.predict_proba(X_ext_scaled)[:, 1])

        log.info("  Baselines: IQR Jac=%.4f, Sparsity Jac=%.4f, DT Jac=%.4f",
                 jac_iqr, jac_spar, jac_dt)

        transfer_results[cancer_abbr] = {
            "auc": auc, "f1": f1, "jaccard": jac, "acc": acc,
            "n_retained_hts": int(y_ext.sum()),
            "n_retained_ml":  int(y_pred.sum()),
            "discordant":     disc,
            "y_pred": y_pred, "y_prob": y_prob, "y_true": y_ext,
            "jac_iqr": jac_iqr, "jac_spar": jac_spar, "jac_dt": jac_dt,
            "auc_iqr": auc_iqr, "auc_spar": auc_spar, "auc_dt": auc_dt,
        }

    return transfer_results, scaler, lgb_model, X_train, y_train


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(transfer_results, results, outdir):
    """Save external_validation_results.csv (matches article Table)."""
    rows = []
    for label, abbr in [("BRCA (test set)", "BRCA"),
                         ("LUAD (transfer)", "LUAD"),
                         ("COAD (transfer)", "COAD")]:
        r = transfer_results[abbr]
        row = {
            "Cancer":             label,
            "n_samples":          results[abbr]["n_samples"],
            "n_genes":            results[abbr]["n_genes"],
            "HTSFilter_retained": r["n_retained_hts"],
            "ML_retained":        r["n_retained_ml"],
            "Discordant":         r["discordant"],
            "AUC":                round(r["auc"], 5),
            "F1":                 round(r["f1"], 4),
            "Accuracy":           round(r["acc"], 4),
            "Jaccard":            round(r["jaccard"], 4),
        }
        if abbr != "BRCA":
            row["IQR_Jaccard"]      = round(r["jac_iqr"], 4)
            row["Sparsity_Jaccard"] = round(r["jac_spar"], 4)
            row["DT_Jaccard"]       = round(r["jac_dt"], 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "external_validation_results.csv")
    df.to_csv(csv_path, index=False)
    log.info("Saved: %s", csv_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — 4-panel cross-cancer validation
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure_10(transfer_results, results, outdir):
    """
    Generate Figure 10 — 4-panel cross-cancer external validation figure.

    Panel A: ROC curves BRCA + LUAD + COAD
    Panel B: Jaccard comparison bar chart (LightGBM vs baselines)
    Panel C: Gene retention rate (HTSFilter vs LightGBM)
    Panel D: AUC/F1/Jaccard summary table
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── Panel A: ROC curves ───────────────────────────────────────────────────
    for abbr, label, ls in [("BRCA", "BRCA (test set)", "-"),
                              ("LUAD", "LUAD (transfer)", "--"),
                              ("COAD", "COAD (transfer)", ":")]:
        r = transfer_results[abbr]
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        ax_a.plot(fpr, tpr, color=COLORS[abbr], lw=2.5, ls=ls,
                  label=f"{label}\n(AUC = {r['auc']:.4f})")

    ax_a.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax_a.set_xlabel("False Positive Rate", fontsize=11)
    ax_a.set_ylabel("True Positive Rate", fontsize=11)
    ax_a.set_title("(A) ROC Curves — Zero-Shot Transfer\n(BRCA-trained → LUAD, COAD)",
                   fontsize=11, fontweight="bold")
    ax_a.legend(fontsize=9, loc="lower right")
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)
    sns.despine(ax=ax_a)

    # ── Panel B: Jaccard comparison ───────────────────────────────────────────
    for ax_idx, (abbr, label) in enumerate([("LUAD", "LUAD"), ("COAD", "COAD")]):
        r = transfer_results[abbr]
        methods  = ["LightGBM", "Decision Tree", "Sparsity", "IQR"]
        jaccards = [r["jaccard"], r["jac_dt"], r["jac_spar"], r["jac_iqr"]]
        colors_b = [COLORS[abbr], "#E9ED4C", "#ECE9E2", "#FAF9F3"]
        x = np.arange(len(methods))
        bars = ax_b.bar(x + ax_idx * (len(methods) + 1),
                        jaccards, color=colors_b, edgecolor="white",
                        linewidth=1.2, width=0.7)
        for bar, val in zip(bars, jaccards):
            ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                      f"{val:.4f}", ha="center", va="bottom",
                      fontsize=8, fontweight="bold")

    ax_b.set_ylabel("Jaccard Similarity (vs. HTSFilter)", fontsize=11)
    ax_b.set_title("(B) Jaccard — LightGBM vs Baselines\n(LUAD and COAD, zero-shot)",
                   fontsize=11, fontweight="bold")
    ax_b.set_ylim(0.4, 1.05)
    ax_b.axhline(0.97, color="gray", ls=":", lw=1.5, label="Jaccard = 0.97")
    ax_b.set_xticks([1.5, 6.5])
    ax_b.set_xticklabels(["LUAD", "COAD"], fontsize=11)
    ax_b.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax_b)

    # ── Panel C: Retention rate ───────────────────────────────────────────────
    cancers_c = ["BRCA\n(test set)", "LUAD\n(transfer)", "COAD\n(transfer)"]
    abbrs_c   = ["BRCA", "LUAD", "COAD"]
    hts_ret   = [transfer_results[a]["n_retained_hts"] / results[a]["n_genes"] * 100
                 for a in abbrs_c]
    ml_ret    = [transfer_results[a]["n_retained_ml"]  / results[a]["n_genes"] * 100
                 for a in abbrs_c]
    x = np.arange(len(cancers_c))
    w = 0.35
    ax_c.bar(x - w/2, hts_ret, w, label="HTSFilter",
             color="#ECE9E2", edgecolor="white")
    ax_c.bar(x + w/2, ml_ret,  w, label="LightGBM",
             color="#0279EE", edgecolor="white")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(cancers_c, fontsize=10)
    ax_c.set_ylabel("Genes Retained (%)", fontsize=11)
    ax_c.set_title("(C) Gene Retention Rate\n(HTSFilter vs LightGBM surrogate)",
                   fontsize=11, fontweight="bold")
    ax_c.set_ylim(0, 110)
    ax_c.legend(fontsize=10)
    ax_c.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax_c)

    # ── Panel D: Summary table ────────────────────────────────────────────────
    summary_data = []
    for abbr, label in [("BRCA", "BRCA (test)"),
                         ("LUAD", "LUAD"),
                         ("COAD", "COAD")]:
        r = transfer_results[abbr]
        summary_data.append([
            label,
            f"{r['auc']:.5f}",
            f"{r['f1']:.4f}",
            f"{r['jaccard']:.4f}",
            f"{r['acc']:.4f}",
            str(r["discordant"]),
        ])

    ax_d.axis("off")
    col_labels = ["Cohort", "AUC", "F1", "Jaccard", "Accuracy", "Discordant"]
    table = ax_d.table(
        cellText=summary_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.2, 1, 0.7],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#000000")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ECE9E2")
    ax_d.set_title(
        "(D) External Validation Summary\n(LightGBM surrogate, zero-shot transfer)",
        fontsize=11, fontweight="bold", pad=20)

    fig.suptitle(
        "Figure 10. Cross-Cancer External Validation of the LightGBM Surrogate "
        "via Zero-Shot Transfer\n"
        "TCGA-BRCA (test set), TCGA-LUAD, TCGA-COAD — recount2 / tcga_matrix.h5",
        fontsize=11, y=1.01,
    )

    plt.tight_layout()
    for ext in ("png", "svg"):
        path = os.path.join(outdir, f"Figure_10_external_validation.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-shot cross-cancer external validation (Article §3.11)"
    )
    p.add_argument("--h5",     default="tcga_matrix.h5",
                   help="Path to tcga_matrix.h5 (default: tcga_matrix.h5)")
    p.add_argument("--outdir", default="results/",
                   help="Output directory (default: results/)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    log.info("=" * 65)
    log.info("External Validation — Zero-Shot Cross-Cancer Transfer")
    log.info("  H5 file : %s", args.h5)
    log.info("  Outdir  : %s", args.outdir)
    log.info("=" * 65)

    # 1. Load data and compute 18 features + HTSFilter labels
    results = load_and_process(args.h5)

    # 2. Train on BRCA, transfer to LUAD and COAD
    transfer_results, scaler, lgb_model, X_train, y_train = \
        run_transfer_evaluation(results)

    # 3. Save CSV
    df = save_results_csv(transfer_results, results, args.outdir)

    # 4. Generate Figure 10
    plot_figure_10(transfer_results, results, args.outdir)

    # 5. Print summary
    log.info("\n%s", "=" * 65)
    log.info("TRANSFER RESULTS SUMMARY (article values)")
    log.info("%s", "=" * 65)
    for label, abbr in [("BRCA (test set)", "BRCA"),
                         ("LUAD (transfer)", "LUAD"),
                         ("COAD (transfer)", "COAD")]:
        r = transfer_results[abbr]
        log.info("\n%s:", label)
        log.info("  AUC     = %.5f", r["auc"])
        log.info("  F1      = %.4f", r["f1"])
        log.info("  Jaccard = %.4f", r["jaccard"])
        log.info("  Acc     = %.4f", r["acc"])
        log.info("  HTSFilter retained : %d", r["n_retained_hts"])
        log.info("  LightGBM retained  : %d", r["n_retained_ml"])
        log.info("  Discordant genes   : %d", r["discordant"])
        if abbr != "BRCA":
            log.info("  Baselines — IQR Jac=%.4f, Sparsity Jac=%.4f, DT Jac=%.4f",
                     r["jac_iqr"], r["jac_spar"], r["jac_dt"])

    print("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
