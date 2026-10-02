"""
Draw a mirrored horizontal bar chart comparing miRNA betweenness centrality
between two networks (e.g. tumor vs normal), using the centrality tables
produced by `network_construction.py`.

miRNAs are plotted with their "condition B" (e.g. tumor) centrality on one
side of a shared zero axis and their "condition A" (e.g. normal) centrality
mirrored on the other side, so hub miRNAs that are specific to one condition
are immediately visible.

Usage
-----
    python -m network_analysis.network_centrality_barplot \
        --condition-a-table results/network_centrality_normal.csv --condition-a-label Normal \
        --condition-b-table results/network_centrality_tumor.csv --condition-b-label Tumor \
        --mirnas hsa-mir-example-1 hsa-mir-example-2 hsa-mir-example-3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config
import utils


def load_centrality(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df.set_index("node")["betweenness_centrality"]


def build_comparison_table(
    series_a: pd.Series, series_b: pd.Series, mirnas: list[str] | None
) -> pd.DataFrame:
    """Combine two centrality series into a two-column table, defaulting missing
    nodes to zero centrality (i.e. absent from that condition's network)."""
    index = mirnas if mirnas else sorted(set(series_a.index) | set(series_b.index))
    table = pd.DataFrame(
        {
            "centrality_a": series_a.reindex(index).fillna(0.0),
            "centrality_b": series_b.reindex(index).fillna(0.0),
        },
        index=index,
    )
    table["max_centrality"] = table[["centrality_a", "centrality_b"]].max(axis=1)
    return table.sort_values("max_centrality", ascending=False).drop(columns="max_centrality")


def plot_mirrored_bars(
    table: pd.DataFrame, label_a: str, label_b: str, output_path: Path
) -> None:
    utils.set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(table) + 1.5))

    ax.barh(table.index, -table["centrality_b"], color="#E78AC3", edgecolor="black", label=f"{label_b} network")
    ax.barh(table.index, table["centrality_a"], color="#8AE7A8", edgecolor="black", label=f"{label_a} network")

    ax.set_xlabel("Betweenness Centrality")
    ax.axvline(0, color="black", linewidth=1.5)
    ax.legend(loc="upper left", frameon=False)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([f"{abs(x):.3f}" for x in ax.get_xticks()])

    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"Figure written to: {output_path}")


def run(
    table_a_path: Path,
    label_a: str,
    table_b_path: Path,
    label_b: str,
    mirnas: list[str] | None,
    output_path: Path,
) -> None:
    series_a = load_centrality(table_a_path)
    series_b = load_centrality(table_b_path)
    table = build_comparison_table(series_a, series_b, mirnas)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_mirrored_bars(table, label_a, label_b, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-a-table", type=Path, required=True)
    parser.add_argument("--condition-a-label", type=str, default="Normal")
    parser.add_argument("--condition-b-table", type=Path, required=True)
    parser.add_argument("--condition-b-label", type=str, default="Tumor")
    parser.add_argument(
        "--mirnas", nargs="+", default=None,
        help="Restrict the plot to these miRNA node names. Defaults to every node "
             "present in either centrality table.",
    )
    parser.add_argument(
        "--output", type=Path, default=config.RESULTS_DIR / "network_centrality_comparison.png"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.condition_a_table, args.condition_a_label,
        args.condition_b_table, args.condition_b_label,
        args.mirnas, args.output,
    )
