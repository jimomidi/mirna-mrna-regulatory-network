"""
Cross-validated XGBoost diagnostic classifier (tumor vs. normal) for a
single-gene expression predictor, with class balancing, hyper-parameter
search, and probability calibration.

For each gene: normal- and tumor-sample expression are filtered to
sufficiently-expressed values, trimmed of IQR outliers, combined, and
standardized. A 5-fold stratified cross-validation loop then, per fold:
balances the training data with SMOTE, tunes an XGBoost classifier with a
randomized hyper-parameter search (optimizing ROC AUC), calibrates its
predicted probabilities, and evaluates on the held-out fold (ROC AUC,
Mann-Whitney U test p-value, and the F1-optimal decision threshold). The
mean ROC curve across folds is plotted per gene.

This mirrors a diagnostic-marker validation workflow; it is computationally
heavier than `roc_diagnostic_analysis.py` and intended for a smaller number
of candidate genes.

Note on batch-effect correction
--------------------------------
An earlier version of this script regressed expression on a normal/tumor
indicator before standardizing, framed as batch-effect correction. That step
has been removed: the "batch" indicator was identical to the normal/tumor
label being classified, so regressing it out and taking residuals removes,
by construction, the between-group mean difference the classifier is meant
to detect. See `differential_expression/expression_boxplot.py` for the same
issue and a fuller explanation. Expression is now IQR-trimmed and
standardized directly, without that step.

Usage
-----
    python -m classification_and_survival.ml_classifier_roc \
        --genes GENE_A GENE_B GENE_C \
        --output results/ml_classifier_roc.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import mannwhitneyu
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
import utils

PARAM_DIST = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5, 6],
    "n_estimators": [50, 100, 150, 200],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "reg_alpha": [0, 0.5, 1],
    "reg_lambda": [0.5, 1, 1.5],
    "gamma": [0, 0.1, 0.3],
}


def _iqr_trim(values: np.ndarray) -> np.ndarray:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    mask = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
    return values[mask]


def prepare_gene_data(
    gene: str, tumor_df: pd.DataFrame, normal_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (X, y) for one gene after low-expression filtering, IQR trimming,
    and standardization, or None if the gene fails the expression filter."""
    if gene not in normal_df.index:
        print(f"  {gene}: not found in normal expression data, skipped")
        return None
    normal_vals = normal_df.loc[gene].values.astype(float)
    if np.sum(normal_vals >= 1) / len(normal_vals) < config.MIN_NONZERO_FRACTION:
        print(f"  {gene}: failed the normal-sample expression filter, skipped")
        return None
    normal_vals = _iqr_trim(normal_vals)

    tumor_vals = tumor_df[tumor_df["gene_name"] == gene]["tpm_unstranded"].astype(float).values
    if tumor_vals.size == 0:
        print(f"  {gene}: not found in tumor expression data, skipped")
        return None
    if np.sum(tumor_vals >= 1) / len(tumor_vals) < config.MIN_NONZERO_FRACTION:
        print(f"  {gene}: failed the tumor-sample expression filter, skipped")
        return None
    tumor_vals = _iqr_trim(tumor_vals)

    all_vals = np.concatenate([normal_vals, tumor_vals])
    all_labels = np.concatenate([np.zeros(len(normal_vals)), np.ones(len(tumor_vals))])

    X_expr = all_vals.reshape(-1, 1)
    X_standardized = (X_expr - np.mean(X_expr)) / np.std(X_expr)

    return X_standardized, all_labels.astype(int)


def cross_validated_roc(X: np.ndarray, y: np.ndarray, gene: str, random_state: int = config.RANDOM_STATE):
    """Run the stratified 5-fold CV loop and return (mean_fpr, mean_tpr, mean_auc, summary)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    aucs, pvals, fprs, tprs, thresholds = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_resampled, y_resampled = SMOTE(random_state=random_state).fit_resample(X_train_scaled, y_train)

        xgb_base = XGBClassifier(random_state=random_state, eval_metric="logloss")
        search = RandomizedSearchCV(
            estimator=xgb_base, param_distributions=PARAM_DIST, n_iter=20,
            scoring="roc_auc", cv=3, verbose=0, n_jobs=1, random_state=random_state,
        )
        search.fit(X_resampled, y_resampled)

        model = CalibratedClassifierCV(estimator=search.best_estimator_, cv=3)
        model.fit(X_resampled, y_resampled)

        X_test_scaled = scaler.transform(X_test)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        aucs.append(auc(fpr, tpr))
        fprs.append(fpr)
        tprs.append(tpr)

        precision, recall, thresh = precision_recall_curve(y_test, y_pred_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        thresholds.append(thresh[np.argmax(f1_scores)] if len(thresh) else np.nan)

        try:
            _, p = mannwhitneyu(
                X_test[y_test == 0].ravel(), X_test[y_test == 1].ravel(),
                alternative="two-sided", method="exact",
            )
        except ValueError:
            p = np.nan
        pvals.append(p)

    print(f"\n{gene} - fold-wise AUC / p-value / best threshold:")
    for i, (a, p, t) in enumerate(zip(aucs, pvals, thresholds), 1):
        print(f"  fold {i}: AUC = {a:.4f}, p = {p:.2e}, threshold = {t:.2f}")
    print(f"  mean AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  mean p-value = {np.nanmean(pvals):.2e}")
    print(f"  mean threshold = {np.nanmean(thresholds):.2f}")

    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    for fpr, tpr in zip(fprs, tprs):
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        interp_tprs.append(interp)
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    return mean_fpr, mean_tpr, mean_auc


def run(genes: list[str], tumor_file: Path, normal_file: Path, output_path: Path) -> None:
    warnings.filterwarnings("ignore")
    utils.set_plot_style()
    plt.rcParams["font.size"] = 20

    tumor_df = utils.load_tumor_expression(tumor_file)
    normal_df = pd.read_csv(normal_file).drop(columns=["Name"]).set_index("Description")

    fig, axes = plt.subplots(1, len(genes), figsize=(6 * len(genes), 6))
    axes = np.atleast_1d(axes)

    for i, gene in enumerate(genes):
        ax = axes[i]
        prepared = prepare_gene_data(gene, tumor_df, normal_df)
        if prepared is None:
            ax.set_visible(False)
            continue
        X, y = prepared
        mean_fpr, mean_tpr, mean_auc = cross_validated_roc(X, y, gene)

        ax.plot(mean_fpr, mean_tpr, color="darkblue", lw=2, label=f"Mean AUC = {mean_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(gene, fontsize=20, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Sensitivity", fontsize=20, fontweight="bold")
        if i == len(genes) - 1:
            ax.set_xlabel("1 - Specificity", fontsize=20, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", nargs="+", default=config.GENES_OF_INTEREST)
    parser.add_argument("--tumor-file", type=Path, default=config.TUMOR_EXPRESSION_FILE)
    parser.add_argument("--normal-file", type=Path, default=config.NORMAL_EXPRESSION_FILE)
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "ml_classifier_roc.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.genes, args.tumor_file, args.normal_file, args.output)
