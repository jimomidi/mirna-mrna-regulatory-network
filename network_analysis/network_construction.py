"""
Build a miRNA-mRNA regulatory network from expression data and a predicted
miRNA-target interaction reference, score edges by expression correlation,
and compute betweenness centrality for every node.

Pipeline
--------
1. Load miRNA and mRNA expression, pivot to (feature x sample) matrices on
   their shared sample set, and log2(x + 1) transform.
2. Load predicted miRNA-target interactions (e.g. concatenated TargetScan /
   miRDB / miRTarBase exports in "Var1, Var2, Freq" long format) and keep
   pairs with prediction weight >= `--min-weight`.
3. For each candidate interaction with both partners present in the
   expression data, compute the Pearson correlation between their expression
   profiles across shared samples and keep the edge if r is negative and
   passes the significance threshold (the expected direction for
   miRNA-mediated repression of its target).
4. Compute betweenness centrality for every node and write it to CSV, then
   draw the network with node size proportional to centrality.

Note
----
An earlier version of this script also attempted to attach one additional
"best-supported" external node to the network, chosen purely by its total
predicted-interaction weight to nodes already in the network. That step has
been removed: it added an edge without checking the same
correlation-sign/significance criterion applied to every other edge in the
network (step 3 above), so it could insert a node that never actually showed
anticorrelated expression with its neighbors - a methodologically
inconsistent, non-statistical shortcut into network membership. Every node
in the network now enters it through the same expression-correlation
filter.

This script is meant to be run once per condition (e.g. tumor vs normal),
pointing `--label` at the condition name, so that
`network_centrality_barplot.py` can compare the resulting centrality tables.

Usage
-----
    python -m network_analysis.network_construction \
        --mirna-file data/expression/tumor_mirna.csv \
        --mrna-file data/expression/tumor_expression.csv \
        --interactions data/interactions/interaction_matrix_tumor.csv \
        --label tumor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import pearsonr

import config
import utils

MIRNA_NODE_COLOR = "#E78AC3"
MRNA_NODE_COLOR = "#8AE7A8"


def load_expression_matrices(mirna_file: Path, mrna_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return log2(count + 1) miRNA and mRNA matrices on their shared sample set."""
    raw_mirna = pd.read_csv(mirna_file)
    raw_mrna = pd.read_csv(mrna_file)

    mirna = raw_mirna.pivot_table(index="miRNA_ID", columns="case_id", values="read_count", aggfunc="sum")
    mrna = raw_mrna.pivot_table(index="gene_name", columns="case_id", values="unstranded", aggfunc="sum")

    common_samples = mirna.columns.intersection(mrna.columns)
    mirna = mirna[common_samples]
    mrna = mrna[common_samples]

    mirna.index = mirna.index.str.lower()
    mrna.index = mrna.index.str.upper()

    mirna = np.log2(mirna + 1)
    mrna = np.log2(mrna + 1)
    return mirna, mrna


def load_interactions(interaction_files: list[Path], min_weight: int) -> pd.DataFrame:
    """Load and concatenate predicted-interaction tables, filtered by weight."""
    frames = [pd.read_csv(f) for f in interaction_files]
    interactions = pd.concat(frames, axis=0, ignore_index=True)
    interactions = interactions.rename(columns={"Var1": "miRNA_ID", "Var2": "gene_id", "Freq": "weight"})
    interactions = interactions[interactions["weight"] >= min_weight].copy()
    interactions["miRNA_ID"] = interactions["miRNA_ID"].str.lower()
    interactions["gene_id"] = interactions["gene_id"].str.upper()
    return interactions


def build_network(
    interactions: pd.DataFrame,
    mirna_expr: pd.DataFrame,
    mrna_expr: pd.DataFrame,
    correlation_threshold: float = config.NETWORK_CORRELATION_THRESHOLD,
    p_threshold: float = config.PVALUE_THRESHOLD,
) -> nx.Graph:
    """Add an edge for every candidate interaction whose expression correlation
    is negative and statistically significant."""
    graph = nx.Graph()
    valid_edges, skipped = 0, 0

    for _, row in interactions.iterrows():
        mirna, gene = row["miRNA_ID"], row["gene_id"]
        if mirna not in mirna_expr.index or gene not in mrna_expr.index:
            skipped += 1
            continue

        expr_mirna = mirna_expr.loc[mirna]
        expr_gene = mrna_expr.loc[gene]
        if expr_mirna.isna().all() or expr_gene.isna().all():
            skipped += 1
            continue
        if expr_mirna.nunique() <= 1 or expr_gene.nunique() <= 1:
            skipped += 1
            continue

        r, p = pearsonr(expr_mirna, expr_gene)
        if r < correlation_threshold and p < p_threshold:
            graph.add_edge(mirna, gene, weight=row["weight"])
            valid_edges += 1

    print(f"Valid interactions retained: {valid_edges} | skipped: {skipped}")
    print(f"Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def draw_network(graph: nx.Graph, centrality: dict, output_path: Path, top_n_labels: int = 5) -> None:
    """Draw the network with node size proportional to betweenness centrality."""
    utils.set_plot_style()

    mirna_nodes = [n for n in graph.nodes() if n.startswith("hsa-mir") or n.startswith("mir")]
    mrna_nodes = [n for n in graph.nodes() if n not in mirna_nodes]

    values = np.array([centrality[n] for n in graph.nodes()])
    min_size, max_size = 300, 2000
    if values.max() > values.min():
        norm = (values - values.min()) / (values.max() - values.min())
    else:
        norm = np.zeros_like(values)
    node_sizes = {n: min_size + s * (max_size - min_size) for n, s in zip(graph.nodes(), norm)}

    plt.figure(figsize=(16, 12))
    pos = nx.kamada_kawai_layout(graph)

    nx.draw_networkx_nodes(
        graph, pos, nodelist=mirna_nodes, node_color=MIRNA_NODE_COLOR, alpha=0.95,
        node_size=[node_sizes[n] for n in mirna_nodes], edgecolors="black", linewidths=0.8,
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=mrna_nodes, node_color=MRNA_NODE_COLOR, alpha=0.95,
        node_size=[node_sizes[n] for n in mrna_nodes], edgecolors="black", linewidths=0.8,
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.8)

    top_mirnas = sorted(
        ((n, centrality[n]) for n in mirna_nodes), key=lambda x: x[1], reverse=True
    )[:top_n_labels]
    for node, _ in top_mirnas:
        x, y = pos[node]
        plt.text(
            x, y, node, fontsize=18, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.2"),
        )

    legend_elements = [
        Patch(facecolor=MIRNA_NODE_COLOR, edgecolor="black", label="miRNA"),
        Patch(facecolor=MRNA_NODE_COLOR, edgecolor="black", label="mRNA"),
    ]
    plt.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.7, 0.95), fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=config.FIGURE_DPI)
    plt.close()
    print(f"Network figure written to: {output_path}")


def run(
    mirna_file: Path,
    mrna_file: Path,
    interaction_files: list[Path],
    label: str,
    min_weight: int,
) -> None:
    output_dir = config.ensure_results_dir()

    mirna_expr, mrna_expr = load_expression_matrices(mirna_file, mrna_file)
    interactions = load_interactions(interaction_files, min_weight)

    graph = build_network(interactions, mirna_expr, mrna_expr)

    if graph.number_of_nodes() == 0:
        print(
            "No edges passed the correlation/significance/weight filters - "
            "the network is empty. No centrality table or figure will be written."
        )
        return

    centrality = nx.betweenness_centrality(graph)
    centrality_df = pd.DataFrame(
        {"node": list(centrality.keys()), "betweenness_centrality": list(centrality.values())}
    ).sort_values("betweenness_centrality", ascending=False)
    centrality_df["node_type"] = np.where(
        centrality_df["node"].astype(str).str.startswith("hsa-mir")
        | centrality_df["node"].astype(str).str.startswith("mir"),
        "miRNA", "mRNA",
    )

    centrality_path = output_dir / f"network_centrality_{label}.csv"
    centrality_df.to_csv(centrality_path, index=False)
    print(f"Centrality table written to: {centrality_path}")

    top_mirnas = centrality_df[centrality_df["node_type"] == "miRNA"].head(10)
    print("\nTop miRNA hub nodes by betweenness centrality:")
    print(top_mirnas.to_string(index=False))

    figure_path = output_dir / f"network_{label}.png"
    draw_network(graph, centrality, figure_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirna-file", type=Path, default=config.TUMOR_MIRNA_FILE)
    parser.add_argument("--mrna-file", type=Path, default=config.TUMOR_EXPRESSION_FILE)
    parser.add_argument(
        "--interactions", nargs="+", type=Path, required=True,
        help="One or more predicted-interaction CSV files (Var1/Var2/Freq format).",
    )
    parser.add_argument(
        "--label", type=str, required=True,
        help="Condition label used to name output files (e.g. 'tumor' or 'normal').",
    )
    parser.add_argument("--min-weight", type=int, default=config.MIN_INTERACTION_WEIGHT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.mirna_file, args.mrna_file, args.interactions, args.label, args.min_weight)
