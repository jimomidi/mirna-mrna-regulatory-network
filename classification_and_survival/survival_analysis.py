"""
Overall-survival analysis: single-gene Kaplan-Meier curves (median-expression
split) plus a multi-gene Cox proportional-hazards risk score, each with a
log-rank test, hazard ratio, and median survival time (95% CI).

Overall survival time is computed from the clinical table as
min(days_to_death, days_to_last_follow_up), converted to months, and the
event indicator is 1 for deceased cases and 0 otherwise (right-censored).

Usage
-----
    python -m classification_and_survival.survival_analysis \
        --genes GENE_A GENE_B \
        --output results/survival_analysis.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import config
import utils

GROUP_COLORS = {"High": "#67001f", "Low": "#053061"}


def build_survival_table(tumor_df: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    clinical = clinical[
        ["case_id", "days_to_death", "days_to_last_follow_up", "vital_status"]
    ].drop_duplicates("case_id").copy()
    clinical["days_to_death"] = pd.to_numeric(clinical["days_to_death"], errors="coerce")
    clinical["days_to_last_follow_up"] = pd.to_numeric(clinical["days_to_last_follow_up"], errors="coerce")
    clinical["OS_time"] = clinical[["days_to_death", "days_to_last_follow_up"]].min(axis=1) / 30.0
    clinical["OS_event"] = clinical["vital_status"].apply(lambda x: 1 if str(x).strip().lower() == "dead" else 0)
    return clinical


def _median_ci(kmf: KaplanMeierFitter) -> tuple[float, float, float]:
    median = kmf.median_survival_time_
    ci = kmf.confidence_interval_
    return median, ci.iloc[:, 0].dropna().min(), ci.iloc[:, 1].dropna().max()


def _plot_km_group(ax, df: pd.DataFrame, group_col: str, title: str, x_max: float) -> None:
    kmf = KaplanMeierFitter()
    median_info = {}
    for label in ["Low", "High"]:
        subset = df[df[group_col] == label]
        kmf.fit(subset["OS_time"], subset["OS_event"], label=label)
        kmf.plot(ax=ax, ci_show=True, color=GROUP_COLORS[label], linewidth=2, legend=False)
        median_info[label] = _median_ci(kmf)

    result = logrank_test(
        df.loc[df[group_col] == "Low", "OS_time"], df.loc[df[group_col] == "High", "OS_time"],
        event_observed_A=df.loc[df[group_col] == "Low", "OS_event"],
        event_observed_B=df.loc[df[group_col] == "High", "OS_event"],
    )
    n_high = (df[group_col] == "High").sum()
    n_low = (df[group_col] == "Low").sum()
    hr = (
        (df.loc[df[group_col] == "High", "OS_event"].sum() / n_high)
        / (df.loc[df[group_col] == "Low", "OS_event"].sum() / n_low)
    )

    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xlabel("Months", fontsize=16, fontweight="bold")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(max(x_max // 4, 1)))
    ax.text(
        0.02, 0.05,
        f"Logrank p = {result.p_value:.1e}\n"
        f"HR (high vs low) = {hr:.2f}\n"
        f"n(high) = {n_high}, n(low) = {n_low}\n"
        f"Median High = {median_info['High'][0]:.1f}m [{median_info['High'][1]:.1f}-{median_info['High'][2]:.1f}]\n"
        f"Median Low = {median_info['Low'][0]:.1f}m [{median_info['Low'][1]:.1f}-{median_info['Low'][2]:.1f}]",
        transform=ax.transAxes, fontsize=11,
    )


def run(genes: list[str], tumor_file: Path, clinical_file: Path, output_path: Path, x_max: float) -> None:
    warnings.filterwarnings("ignore")
    utils.set_plot_style()
    plt.rcParams["font.size"] = 12

    tumor_df = utils.load_tumor_expression(tumor_file)
    clinical = build_survival_table(tumor_df, pd.read_csv(clinical_file))

    fig, axes = plt.subplots(1, len(genes) + 1, figsize=(6 * (len(genes) + 1), 6))
    axes = np.atleast_1d(axes)

    for idx, gene in enumerate(genes):
        expr = tumor_df[tumor_df["gene_name"] == gene][["case_id", "fpkm_unstranded"]].copy()
        expr["log2_fpkm"] = utils.log2_transform(expr["fpkm_unstranded"])
        df = pd.merge(expr, clinical, on="case_id").dropna(subset=["OS_time", "OS_event", "log2_fpkm"])

        median_expr = df["log2_fpkm"].median()
        df["group"] = np.where(df["log2_fpkm"] >= median_expr, "High", "Low")
        _plot_km_group(axes[idx], df, "group", gene, x_max)
        if idx == 0:
            axes[idx].set_ylabel("Percent Survival", fontsize=16, fontweight="bold")

    expr_pivot = tumor_df[tumor_df["gene_name"].isin(genes)].pivot_table(
        index="case_id", columns="gene_name", values="fpkm_unstranded", aggfunc="mean"
    )
    expr_pivot = utils.log2_transform(expr_pivot)
    df = pd.merge(clinical, expr_pivot, left_on="case_id", right_index=True)
    df = df.dropna(subset=["OS_time", "OS_event"] + genes)

    cph = CoxPHFitter()
    cph.fit(df[["OS_time", "OS_event"] + genes], duration_col="OS_time", event_col="OS_event")
    df["risk_score"] = cph.predict_partial_hazard(df[genes])
    median_risk = df["risk_score"].median()
    df["risk_group"] = np.where(df["risk_score"] >= median_risk, "High", "Low")

    _plot_km_group(axes[-1], df, "risk_group", "Multi-gene Risk Model", x_max)

    legend_lines = [Line2D([0], [0], color=GROUP_COLORS["Low"], lw=2), Line2D([0], [0], color=GROUP_COLORS["High"], lw=2)]
    axes[-1].legend(legend_lines, ["Low Group", "High Group"], loc="center left",
                     bbox_to_anchor=(1.01, 0.65), fontsize=12, frameon=False)

    print("\nCox proportional-hazards model summary:")
    print(cph.summary[["coef", "exp(coef)", "p"]])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", nargs="+", default=config.GENES_OF_INTEREST)
    parser.add_argument("--tumor-file", type=Path, default=config.TUMOR_EXPRESSION_FILE)
    parser.add_argument("--clinical-file", type=Path, default=config.CLINICAL_FILE)
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "survival_analysis.png")
    parser.add_argument("--x-max", type=float, default=160, help="Upper x-axis limit, in months.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.genes, args.tumor_file, args.clinical_file, args.output, args.x_max)
