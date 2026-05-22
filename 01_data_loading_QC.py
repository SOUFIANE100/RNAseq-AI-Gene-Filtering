"""
01_data_loading_QC.py
=====================
Load TCGA-BRCA RNA-seq count matrix and HTSFilter labels,
perform quality-control checks, and save a clean merged dataset.

Inputs
------
- data/TCGA_BRCA_counts.csv   : raw count matrix (genes x samples)
- data/HTSFilter_labels.csv   : binary keep/remove labels per gene

Outputs
-------
- data/counts_QC.csv          : QC-passed count matrix
- data/labels_QC.csv          : aligned labels
- figures/QC_summary.png      : library-size distribution + label balance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.2)
Path("figures").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading count matrix …")
counts = pd.read_csv("data/TCGA_BRCA_counts.csv", index_col=0)
labels = pd.read_csv("data/HTSFilter_labels.csv", index_col=0)

print(f"  Counts shape : {counts.shape}  (genes × samples)")
print(f"  Labels shape : {labels.shape}")

# ── 2. Align genes ────────────────────────────────────────────────────────────
common_genes = counts.index.intersection(labels.index)
counts = counts.loc[common_genes]
labels = labels.loc[common_genes]
print(f"  Common genes : {len(common_genes):,}")

# ── 3. Basic QC ───────────────────────────────────────────────────────────────
# Remove genes with zero counts across all samples
nonzero_mask = (counts > 0).any(axis=1)
counts = counts[nonzero_mask]
labels = labels[nonzero_mask]
print(f"  After removing all-zero genes : {counts.shape[0]:,}")

# Library sizes
lib_sizes = counts.sum(axis=0)
print(f"  Library size  min={lib_sizes.min():,.0f}  "
      f"median={lib_sizes.median():,.0f}  max={lib_sizes.max():,.0f}")

# Label balance
n_keep   = (labels.iloc[:, 0] == 1).sum()
n_remove = (labels.iloc[:, 0] == 0).sum()
print(f"  HTSFilter labels : {n_keep:,} keep  |  {n_remove:,} remove")

# ── 4. QC figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(np.log10(lib_sizes + 1), bins=40, color="#0279EE", edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("log₁₀(Library size)")
axes[0].set_ylabel("Number of samples")
axes[0].set_title("Library-size distribution")

axes[1].bar(["Keep (1)", "Remove (0)"], [n_keep, n_remove],
            color=["#75A025", "#FF9400"], edgecolor="white")
axes[1].set_ylabel("Number of genes")
axes[1].set_title("HTSFilter label balance")
for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig("figures/QC_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/QC_summary.png")

# ── 5. Save clean data ────────────────────────────────────────────────────────
counts.to_csv("data/counts_QC.csv")
labels.to_csv("data/labels_QC.csv")
print("  Saved: data/counts_QC.csv")
print("  Saved: data/labels_QC.csv")
print("\nDone.")
