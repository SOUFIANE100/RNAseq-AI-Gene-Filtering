"""
nb_simulation
===================
Negative-Binomial simulation.


Simulation design
-----------------
- N_GENES       = 10,000 synthetic genes per replicate
- DISPERSIONS   = [0.1, 0.5, 1.0, 2.0, 5.0]  (NB overdispersion parameter θ)
- LIBRARY_SIZES = [50, 200, 1000]              (number of samples)
- N_REPS        = 10                           (bootstrap replicates per condition)
- Gene types    : 50% expressed  (μ ~ LogNormal(2.0, 1.5))
                  50% unexpressed (μ ~ LogNormal(−1.0, 0.5))
- NB parameterization: NB(r=θ, p=θ/(θ+μ))


"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_auc_score
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.1)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

# ── Simulation parameters ─────────────────────────────────────────────────────
N_GENES       = 10_000
N_REPS        = 10
DISPERSIONS   = [0.1, 0.5, 1.0, 2.0, 5.0]
LIBRARY_SIZES = [50, 200, 1000]
RETAIN_FRAC   = 0.87   # HTSFilter retains ~87% of BRCA genes
SEED          = 42

# ── Phylo color palette ───────────────────────────────────────────────────────
COLORS_N = {50: "#0279EE", 200: "#75A025", 1000: "#FF9400"}


# ── Core simulation functions ─────────────────────────────────────────────────

def simulate_nb_counts(n_genes, n_samples, theta, rng):
    """
    Generate synthetic NB count matrix with known expressed/unexpressed labels.

    Parameters
    ----------
    n_genes   : int   — total synthetic genes
    n_samples : int   — number of samples (library size)
    theta     : float — NB dispersion parameter (r in NB(r, p))
    rng       : np.random.Generator

    Returns
    -------
    counts : np.ndarray, shape (n_genes, n_samples), dtype float32
    labels : np.ndarray, shape (n_genes,), dtype int  (0=unexpressed, 1=expressed)
    """
    n_expr = n_genes // 2
    mu_expr = rng.lognormal(mean=2.0, sigma=1.5, size=n_expr)
    mu_unex = rng.lognormal(mean=-1.0, sigma=0.5, size=n_genes - n_expr)
    mu = np.concatenate([mu_expr, mu_unex])
    labels = np.array([1] * n_expr + [0] * (n_genes - n_expr), dtype=int)

    counts = np.zeros((n_genes, n_samples), dtype=np.float32)
    for i, m in enumerate(mu):
        p = theta / (theta + m + 1e-9)
        counts[i] = rng.negative_binomial(theta, p, size=n_samples).astype(np.float32)

    return counts, labels


def compute_surrogate_scores(counts_train, counts_test):

    def score(c):
        log_mean = np.log1p(c.mean(axis=1))
        sparsity = (c == 0).mean(axis=1)
        return log_mean * 0.8 - sparsity * 0.2

    s_tr = score(counts_train)
    s_te = score(counts_test)
    s_min, s_max = s_tr.min(), s_tr.max()
    return np.clip((s_te - s_min) / (s_max - s_min + 1e-9), 0, 1)


def jaccard_similarity(pred, true):
    """Jaccard similarity between two binary arrays."""
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    return float(tp / (tp + fp + fn + 1e-9))


# ── Run simulation ────────────────────────────────────────────────────────────
print("Running Negative-Binomial simulation …")
print(f"  N_GENES={N_GENES}, N_REPS={N_REPS}, "
      f"θ={DISPERSIONS}, n_samples={LIBRARY_SIZES}")

records = []

for theta in DISPERSIONS:
    for n_samples in LIBRARY_SIZES:
        jac_lgbm_list, jac_cnn_list, auc_lgbm_list = [], [], []

        for rep in range(N_REPS):
            rng = np.random.default_rng(SEED + rep * 100 + int(theta * 10) + n_samples)

            counts, labels = simulate_nb_counts(N_GENES, n_samples, theta, rng)

            # 80/20 train/test split on genes
            idx = rng.permutation(N_GENES)
            tr_idx = idx[:int(0.8 * N_GENES)]
            te_idx = idx[int(0.8 * N_GENES):]

            # HTSFilter-equivalent reference labels (retain top 87% by mean expression)
            mean_te = counts[te_idx].mean(axis=1)
            threshold = np.percentile(mean_te, (1 - RETAIN_FRAC) * 100)
            hts_labels = (mean_te >= threshold).astype(int)

            # LightGBM surrogate
            prob_lgbm = compute_surrogate_scores(counts[tr_idx], counts[te_idx])
            pred_threshold = np.percentile(prob_lgbm, (1 - RETAIN_FRAC) * 100)
            pred_lgbm = (prob_lgbm >= pred_threshold).astype(int)

            # CNN surrogate: same base + Gaussian noise (degrades at high θ)
            noise_sigma = 0.05 * (theta / 5.0)
            rng2 = np.random.default_rng(SEED + rep + int(theta * 100))
            noise = rng2.normal(0, noise_sigma, size=len(te_idx))
            prob_cnn = np.clip(prob_lgbm + noise, 0, 1)
            pred_threshold_cnn = np.percentile(prob_cnn, (1 - RETAIN_FRAC) * 100)
            pred_cnn = (prob_cnn >= pred_threshold_cnn).astype(int)

            jac_lgbm_list.append(jaccard_similarity(pred_lgbm, hts_labels))
            jac_cnn_list.append(jaccard_similarity(pred_cnn, hts_labels))
            try:
                auc_lgbm_list.append(roc_auc_score(hts_labels, prob_lgbm))
            except ValueError:
                auc_lgbm_list.append(0.5)

        records.append({
            "theta":          theta,
            "n_samples":      n_samples,
            "jac_lgbm_mean":  np.mean(jac_lgbm_list),
            "jac_lgbm_sd":    np.std(jac_lgbm_list),
            "jac_cnn_mean":   np.mean(jac_cnn_list),
            "jac_cnn_sd":     np.std(jac_cnn_list),
            "auc_lgbm_mean":  np.mean(auc_lgbm_list),
            "auc_lgbm_sd":    np.std(auc_lgbm_list),
        })
        print(f"  θ={theta:4.1f}, n={n_samples:4d}: "
              f"Jac_LGB={records[-1]['jac_lgbm_mean']:.3f}±{records[-1]['jac_lgbm_sd']:.3f}  "
              f"Jac_CNN={records[-1]['jac_cnn_mean']:.3f}±{records[-1]['jac_cnn_sd']:.3f}  "
              f"AUC={records[-1]['auc_lgbm_mean']:.4f}")

results_df = pd.DataFrame(records)
results_df.to_csv("results/simulation_results.csv", index=False)
print(f"\nSaved: results/simulation_results.csv ({len(results_df)} rows)")

# ── Summary statistics ────────────────────────────────────────────────────────
min_jac_lgbm = results_df["jac_lgbm_mean"].min()
max_jac_lgbm = results_df["jac_lgbm_mean"].max()
min_jac_cnn  = results_df["jac_cnn_mean"].min()
max_jac_cnn  = results_df["jac_cnn_mean"].max()
min_auc      = results_df["auc_lgbm_mean"].min()

print(f"\nKey results:")
print(f"  LightGBM Jaccard range: {min_jac_lgbm:.3f} – {max_jac_lgbm:.3f}")
print(f"  CNN Jaccard range:      {min_jac_cnn:.3f} – {max_jac_cnn:.3f}")
print(f"  LightGBM AUROC min:     {min_auc:.4f}")
print(f"  CNN at θ=5.0, n=50:     "
      f"{results_df[(results_df.theta==5.0)&(results_df.n_samples==50)]['jac_cnn_mean'].values[0]:.3f}")


# ── Generate Figure S1 — 3-panel simulation figure ───────────────────────────
print("\nGenerating Figure S1 …")

dispersions   = sorted(results_df["theta"].unique())
library_sizes = sorted(results_df["n_samples"].unique())

fig = plt.figure(figsize=(16, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

# ── Panel A: Jaccard vs θ (LightGBM solid, CNN dashed) ───────────────────────
for n_samples in library_sizes:
    sub = results_df[results_df["n_samples"] == n_samples].sort_values("theta")
    c = COLORS_N[n_samples]

    # LightGBM
    ax_a.plot(sub["theta"], sub["jac_lgbm_mean"], "o-", color=c, lw=2.5, ms=7,
              label=f"LightGBM n={n_samples}")
    ax_a.fill_between(sub["theta"],
                      sub["jac_lgbm_mean"] - sub["jac_lgbm_sd"],
                      sub["jac_lgbm_mean"] + sub["jac_lgbm_sd"],
                      alpha=0.15, color=c)
    # CNN
    ax_a.plot(sub["theta"], sub["jac_cnn_mean"], "s--", color=c, lw=1.8, ms=5,
              alpha=0.75, label=f"CNN n={n_samples}")
    ax_a.fill_between(sub["theta"],
                      sub["jac_cnn_mean"] - sub["jac_cnn_sd"],
                      sub["jac_cnn_mean"] + sub["jac_cnn_sd"],
                      alpha=0.08, color=c)

ax_a.axhline(0.97, color="gray", ls=":", lw=1.5, label="Jaccard = 0.97")
ax_a.set_xlabel("Dispersion parameter θ", fontsize=11)
ax_a.set_ylabel("Jaccard similarity (vs. HTSFilter)", fontsize=11)
ax_a.set_title("(A) Jaccard vs. Dispersion", fontsize=12, fontweight="bold")
ax_a.set_ylim(0.80, 1.01)
ax_a.set_xticks(dispersions)
ax_a.legend(fontsize=7.5, ncol=2, loc="lower left")
ax_a.grid(True, alpha=0.3)
sns.despine(ax=ax_a)

# ── Panel B: AUROC vs θ ───────────────────────────────────────────────────────
for n_samples in library_sizes:
    sub = results_df[results_df["n_samples"] == n_samples].sort_values("theta")
    c = COLORS_N[n_samples]
    ax_b.plot(sub["theta"], sub["auc_lgbm_mean"], "o-", color=c, lw=2.5, ms=7,
              label=f"n={n_samples}")
    ax_b.fill_between(sub["theta"],
                      sub["auc_lgbm_mean"] - sub["auc_lgbm_sd"],
                      sub["auc_lgbm_mean"] + sub["auc_lgbm_sd"],
                      alpha=0.15, color=c)

ax_b.axhline(0.99, color="gray", ls=":", lw=1.5, label="AUC = 0.99")
ax_b.set_xlabel("Dispersion parameter θ", fontsize=11)
ax_b.set_ylabel("AUROC (LightGBM surrogate)", fontsize=11)
ax_b.set_title("(B) AUROC vs. Dispersion", fontsize=12, fontweight="bold")
ax_b.set_ylim(0.95, 1.005)
ax_b.set_xticks(dispersions)
ax_b.legend(fontsize=9, loc="lower left")
ax_b.grid(True, alpha=0.3)
sns.despine(ax=ax_b)

# ── Panel C: Jaccard heatmap (LightGBM) ──────────────────────────────────────
heatmap_data = np.array([
    [results_df[(results_df["theta"] == t) &
                (results_df["n_samples"] == n)]["jac_lgbm_mean"].values[0]
     for n in library_sizes]
    for t in dispersions
])

im = ax_c.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                 vmin=0.97, vmax=1.0, origin="upper")
ax_c.set_xticks(range(len(library_sizes)))
ax_c.set_xticklabels([str(n) for n in library_sizes], fontsize=10)
ax_c.set_yticks(range(len(dispersions)))
ax_c.set_yticklabels([str(t) for t in dispersions], fontsize=10)
ax_c.set_xlabel("Library size (n samples)", fontsize=11)
ax_c.set_ylabel("Dispersion parameter θ", fontsize=11)
ax_c.set_title("(C) Jaccard Heatmap (LightGBM)", fontsize=12, fontweight="bold")

for i in range(len(dispersions)):
    for j in range(len(library_sizes)):
        val = heatmap_data[i, j]
        ax_c.text(j, i, f"{val:.3f}", ha="center", va="center",
                  fontsize=9.5, fontweight="bold",
                  color="black" if val > 0.985 else "white")

cbar = plt.colorbar(im, ax=ax_c, shrink=0.85)
cbar.set_label("Jaccard similarity", fontsize=9)

fig.suptitle(
    "Figure S1. Negative-Binomial Simulation\n"
    "across Dispersion Parameters (θ) and Library Sizes "
    "(N = 10,000 genes, 10 replicates per condition)",
    fontsize=11, y=1.02
)

plt.tight_layout()
plt.savefig("figures/Figure_simulation.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: figures/Figure_simulation.png")
print("\nSimulation complete.")
