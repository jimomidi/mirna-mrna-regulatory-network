# miRNA-mRNA Regulatory Network Analysis

A modular analysis pipeline for studying miRNA-mediated gene regulation in
tumor vs. normal expression data: predicted-target filtering, expression
correlation, regulatory network construction, differential expression,
diagnostic classification, survival analysis, and clinical-subgroup
comparisons.

## Purpose

Given tumor RNA-seq/miRNA-seq expression data, a normal/reference-tissue
expression cohort, predicted miRNA-target interactions, and per-case
clinical metadata, this repository reproduces a full candidate-biomarker
workflow:

1. Narrow a predicted miRNA-target interaction database to the targets
   supported by expression anti-correlation.
2. Assemble those interactions into a miRNA-mRNA regulatory network and
   rank nodes by betweenness centrality to identify regulatory hubs.
3. Characterize the differential expression, diagnostic value, and
   prognostic value of the resulting genes across the tumor/normal and
   treatment-defined case groups.

The scripts were generalized from an exploratory, notebook-based analysis;
see [Reproducibility](#reproducibility) for what changed in the process.

## Repository layout

```
config.py                              Central paths, thresholds, and default gene/miRNA lists
utils.py                               Shared data-loading and preprocessing helpers
preprocessing/
    gene_set_overlap.py                 Pairwise/multi-way overlap between gene-symbol lists
    mirna_target_prediction_filter.py   Filter a target-prediction export by score threshold
    mirna_mrna_correlation_filter.py    Correlation-based target filtering/validation
network_analysis/
    network_construction.py             Build the regulatory network, compute centrality
    network_centrality_barplot.py       Compare miRNA hub centrality between two networks
differential_expression/
    volcano_plot.py                     Volcano plot, optionally with genes of interest highlighted
    expression_boxplot.py               Normal vs. tumor expression box/violin plots
    gene_gene_correlation_scatter.py    Pairwise gene-gene expression correlation scatter plots
    correlation_heatmap.py              Normal / tumor / difference correlation heatmaps
    pan_cancer_expression_barplot.py    Pan-cancer log2 fold-change bar chart
classification_and_survival/
    roc_diagnostic_analysis.py          Single- and multi-gene diagnostic ROC curves
    ml_classifier_roc.py                Cross-validated XGBoost diagnostic classifier
    survival_analysis.py                Kaplan-Meier and Cox proportional-hazards survival analysis
clinical_analysis/
    treatment_group_heatmap.py          Expression heatmap across treatment-defined case groups
    clinical_characteristics_heatmap.py Per-case clinical annotation heatmap
```

Each script is self-contained, runnable independently as
`python -m <package>.<script> [options]`, and writes its output figure
(and, where relevant, an underlying results table) to `results/`.

## Requirements

* Python 3.10+
* Packages listed in `requirements.txt`

### Installation

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Expected input data

Place your own data under `data/` (this folder is git-ignored; the layout
below is also documented in `config.py`):

| File | Format | Used by |
| --- | --- | --- |
| `data/expression/tumor_expression.csv` | Long format: `gene_id, gene_name, case_id, tpm_unstranded, fpkm_unstranded, unstranded, ...` (TCGA STAR-Counts style) | all scripts using tumor expression |
| `data/expression/normal_expression.csv` | Wide format: gene identifier column (`Name`, `Description`) followed by one column per sample (GTEx-style export) | all scripts comparing to normal tissue |
| `data/expression/tumor_mirna.csv` | Long format: `miRNA_ID, case_id, read_count, reads_per_million_miRNA_mapped` | `mirna_mrna_correlation_filter.py`, `network_construction.py` |
| `data/metadata/clinical.csv` | One row per case (or per case/treatment): `case_id, age_at_diagnosis, gender, ajcc_pathologic_stage, vital_status, days_to_death, days_to_last_follow_up, treatment_or_therapy, treatment_type` | `survival_analysis.py`, `clinical_analysis/*` |
| `data/interactions/interaction_matrix_*.csv` | `Var1` (miRNA), `Var2` (gene), `Freq` (predicted-interaction weight) - e.g. exported from TargetScan / miRDB / miRTarBase | `network_construction.py` |
| Gene-symbol list CSVs (one `gene` column) | Arbitrary | `gene_set_overlap.py` |

None of these files are bundled with this repository - it is a portable
analysis codebase, not a dataset.

## Usage

Every script accepts `--help` for its full option list. A minimal run of the
full pipeline looks like:

```bash
# 1. Filter predicted targets and validate them by expression correlation
python -m preprocessing.mirna_target_prediction_filter \
    --input data/interactions/targetscan_predictions.csv --mirna hsa-mir-example-1 \
    --output results/mirna_example_1_targets.csv
python -m preprocessing.mirna_mrna_correlation_filter \
    --mirna hsa-mir-example-1 --candidate-genes results/mirna_example_1_targets.csv

# 2. Build the regulatory network and rank hub genes/miRNAs
python -m network_analysis.network_construction \
    --interactions data/interactions/interaction_matrix_tumor.csv --label tumor

# 3. Characterize candidate genes
python -m differential_expression.volcano_plot --input data/differential_expression/results.csv
python -m differential_expression.expression_boxplot --genes GENE_A GENE_B
python -m classification_and_survival.roc_diagnostic_analysis --genes GENE_A GENE_B
python -m classification_and_survival.survival_analysis --genes GENE_A GENE_B
```

Gene and miRNA symbols default to the placeholders in `config.py`
(`GENES_OF_INTEREST`, `MIRNAS_OF_INTEREST`); pass `--genes` / `--mirna`
explicitly, or edit `config.py`, to point the pipeline at your own markers.

## Outputs

Each script writes one or more PNG figures to `results/` (the directory is
created automatically), and several also write a companion CSV of the
underlying computed values (e.g. differential expression statistics,
centrality tables, pairwise overlap tables) so figures can be regenerated or
restyled without rerunning the analysis.

## Reproducibility

* All preprocessing thresholds, statistical tests, and model
  hyperparameters (low-expression filters, correlation and significance
  cutoffs, the XGBoost hyperparameter search grid, cross-validation folds,
  and the `random_state = 42` seed used throughout) are centralized in
  `config.py` and unchanged from the original analysis.
* All file paths are relative to the repository root and configurable via
  command-line arguments; no machine-specific paths remain.
* Case-group definitions for the clinical/treatment comparisons (e.g.
  "received radiotherapy in addition to drug therapy" vs. "drug therapy
  only") are derived programmatically from the clinical table's
  `treatment_type` field rather than pasted-in case ID lists, so the
  analysis reproduces on any cohort with the same clinical schema instead
  of depending on one hard-coded case list.
* `network_centrality_barplot.py` compares two centrality tables produced
  by running `network_construction.py` once per condition (e.g. tumor and
  normal), rather than a fixed table of previously computed centrality
  values, so the comparison is regenerated from the input data.

### Corrections made during a later methodology audit

Two issues were identified and corrected after the initial generalization
pass, both changing analysis behavior rather than just code style:

* **Network construction** (`network_analysis/network_construction.py`)
  no longer attaches an extra "best-supported" external node to the network
  by predicted-interaction weight alone. That step added a node/edge without
  requiring it to pass the same expression-correlation and significance
  filter applied to every other edge, making it a non-statistical, ad hoc
  way for a node to enter the network. Every node now enters the network
  through the same correlation-based filter.
* **Batch-effect correction** was removed from
  `differential_expression/expression_boxplot.py` and
  `classification_and_survival/ml_classifier_roc.py`. Both regressed
  expression on a "batch" indicator that was, in this two-cohort design,
  identical to the normal/tumor label being compared or classified (every
  normal sample from one cohort, every tumor sample from the other), so the
  correction removed the between-group mean difference by construction
  rather than any genuine nuisance variation. Both scripts now use raw
  (log2-transformed) expression directly. This does not apply to
  `network_analysis/network_construction.py`, `roc_diagnostic_analysis.py`,
  or any other script - none of them performed this correction.

## Customization

* Edit `GENES_OF_INTEREST` / `MIRNAS_OF_INTEREST` in `config.py`, or pass
  `--genes` / `--mirna` on the command line, to analyze different markers.
* Edit the thresholds in `config.py` (expression detection threshold,
  correlation/significance cutoffs, minimum interaction weight, random
  seed) to match a different dataset or a more/less stringent analysis.
* `utils.py` centralizes the expression-loading and low-expression
  filtering logic reused across scripts - extend it if you add new input
  formats.

## Notes on scope

This repository intentionally omits a few analyses from the original
exploratory work that either duplicated a more refined version already
included, depended on ad hoc intermediate files with no documented
provenance, or were one-off descriptive/illustrative snippets rather than
reusable pipeline steps. See the accompanying delivery notes for details.
