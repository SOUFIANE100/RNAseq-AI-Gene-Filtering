"""
09_coexpression_network.py
==========================
Build gene co-expression networks for HTSFilter and RF gene sets.
Compute Pearson correlation, apply threshold, detect Louvain modules,
identify hub genes, and compare network topologies.

Parameters
----------
- Top 1,000 most variable genes per gene set
- Pearson |r| > 0.70 edge threshold
- Louvain community detection (resolution=1.0)


"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path

sns.set_theme(style="ticks", font_scale=1.2)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

CORR_THRESH = 0.70
TOP_N_GENES = 1000

# ── 1. Load ───────────────────────────────────────────────────────────────────
counts   = pd.read_csv("data/counts_QC.csv", index_col=0)
features = pd.read_csv("data/gene_features_16.csv", index_col=0)

hts_genes = features.index[features["HTSFilter_label"] == 1].tolist()
rf_genes  = features.index[features.get("pred_RF", features["HTSFilter_label"]) == 1].tolist()

# ── 2. Select top variable genes ─────────────────────────────────────────────
def top_variable(gene_list, counts, n=TOP_N_GENES):
    sub = counts.loc[[g for g in gene_list if g in counts.index]]
    var = sub.var(axis=1).sort_values(ascending=False)
    return var.head(n).index.tolist()

hts_top = top_variable(hts_genes, counts)
rf_top  = top_variable(rf_genes,  counts)
print(f"Top variable genes — HTSFilter: {len(hts_top)}, RF: {len(rf_top)}")

# ── 3. Build network ──────────────────────────────────────────────────────────
def build_network(gene_list, counts, threshold=CORR_THRESH):
    sub = counts.loc[gene_list].T
    corr = sub.corr(method="pearson")
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    for i, g1 in enumerate(gene_list):
        for j, g2 in enumerate(gene_list):
            if j <= i:
                continue
            r = corr.loc[g1, g2]
            if abs(r) >= threshold:
                G.add_edge(g1, g2, weight=abs(r))
    return G, corr

print("Building HTSFilter network …")
G_hts, corr_hts = build_network(hts_top, counts)
print(f"  Nodes: {G_hts.number_of_nodes()}, Edges: {G_hts.number_of_edges()}")

print("Building RF network …")
G_rf, corr_rf = build_network(rf_top, counts)
print(f"  Nodes: {G_rf.number_of_nodes()}, Edges: {G_rf.number_of_edges()}")

# ── 4. Louvain community detection ────────────────────────────────────────────
try:
    import community as community_louvain
    partition_hts = community_louvain.best_partition(G_hts, random_state=42)
    partition_rf  = community_louvain.best_partition(G_rf,  random_state=42)
except ImportError:
    # Fallback: connected components as communities
    partition_hts = {n: i for i, comp in enumerate(nx.connected_components(G_hts)) for n in comp}
    partition_rf  = {n: i for i, comp in enumerate(nx.connected_components(G_rf))  for n in comp}

n_modules_hts = len(set(partition_hts.values()))
n_modules_rf  = len(set(partition_rf.values()))
print(f"Louvain modules — HTSFilter: {n_modules_hts}, RF: {n_modules_rf}")

# ── 5. Hub genes ──────────────────────────────────────────────────────────────
def get_hub_genes(G, top_n=20):
    deg = dict(G.degree())
    return pd.Series(deg).sort_values(ascending=False).head(top_n)

hubs_hts = get_hub_genes(G_hts)
hubs_rf  = get_hub_genes(G_rf)
print(f"Top hub (HTSFilter): {hubs_hts.index[0]} (degree={hubs_hts.iloc[0]})")
print(f"Top hub (RF)       : {hubs_rf.index[0]} (degree={hubs_rf.iloc[0]})")

# ── 6. Network statistics ─────────────────────────────────────────────────────
def net_stats(G, partition, label):
    lcc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc)
    try:
        mod = nx.community.modularity(G, [{n for n,c in partition.items() if c==k}
                                           for k in set(partition.values())])
    except Exception:
        mod = np.nan
    return {
        "Method":       label,
        "Nodes":        G.number_of_nodes(),
        "Edges":        G.number_of_edges(),
        "Density":      round(nx.density(G), 5),
        "LCC_size":     len(lcc),
        "N_modules":    len(set(partition.values())),
        "Modularity":   round(mod, 4) if not np.isnan(mod) else np.nan,
        "Avg_degree":   round(np.mean([d for _, d in G.degree()]), 2),
        "Max_degree":   max(dict(G.degree()).values()),
        "Top_hub":      hubs_hts.index[0] if label == "HTSFilter" else hubs_rf.index[0],
    }

stats_df = pd.DataFrame([
    net_stats(G_hts, partition_hts, "HTSFilter"),
    net_stats(G_rf,  partition_rf,  "RF"),
])
stats_df.to_csv("results/network_statistics_comparison.csv", index=False)
print("\nNetwork statistics:")
print(stats_df.to_string(index=False))

# ── 7. Save edge/node lists ───────────────────────────────────────────────────
for G, part, label in [(G_hts, partition_hts, "HTSFilter"), (G_rf, partition_rf, "RF")]:
    edges = pd.DataFrame([(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
                          columns=["source", "target", "weight"])
    edges.to_csv(f"results/network_edges_{label}.csv", index=False)
    nodes = pd.DataFrame({"gene": list(G.nodes()),
                           "degree": [G.degree(n) for n in G.nodes()],
                           "module": [part.get(n, -1) for n in G.nodes()]})
    nodes.to_csv(f"results/network_nodes_{label}.csv", index=False)
    print(f"  Saved: network_edges_{label}.csv, network_nodes_{label}.csv")

# ── 8. Hub gene network figure ────────────────────────────────────────────────
top_hubs = hubs_hts.head(30).index.tolist()
G_sub = G_hts.subgraph(top_hubs)
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G_sub, seed=42, k=2)
degrees = dict(G_sub.degree())
node_sizes = [degrees[n] * 30 for n in G_sub.nodes()]
modules = [partition_hts.get(n, 0) for n in G_sub.nodes()]
nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, node_color=modules,
                        cmap=cm.Set2, alpha=0.9, ax=ax)
nx.draw_networkx_edges(G_sub, pos, alpha=0.3, width=0.8, ax=ax)
nx.draw_networkx_labels(G_sub, pos, font_size=7, ax=ax)
ax.set_title("Hub Gene Co-expression Network (HTSFilter, top 30 hubs)")
ax.axis("off")
plt.tight_layout()
plt.savefig("figures/hub_gene_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/hub_gene_network.png")

print("\nDone.")
