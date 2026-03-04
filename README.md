# PPI-Characterization

Machine learning-based classification and unsupervised characterization of protein–protein interfaces (PPIs), distinguishing **transmembrane** from **soluble** interfaces. The study compares two feature representations. On the one hand, aggregated physicochemical features and on the other hand Graph Neural Network (GNN) embeddings. Structures are derived from experimentally validated PPIs predicted by AlphaFold3 and refined through molecular dynamics simulations.

This repository contains all code and intermediate data required to reproduce the results of the paper.

 **Paper link will follow.**

---

## Installation

We recommend using a conda environment with Python 3.10 or higher.

```bash
conda create -n slc-env python=3.10
conda activate slc-env

pip install numpy pandas matplotlib scipy statsmodels seaborn scikit-learn ray
```

> **Note:** `ray` is used for parallelised VBGMM fitting. If you do not intend to re-run clustering from scratch, it is not required for the analysis notebooks.

---

## Repository Structure

```
SLC-characterization/
├── source/          # Input CSV files: feature matrices, GNN embeddings, label files,
│                    #                  feature name tables and abbreviation mappings
├── ppi.lib/         # Core Python libraries
│   ├── ml_lib.py                  # Supervised ML utilities: cross-validation, scoring, SVC/GPC/RF wrappers
│   ├── evalmetrics.py             # Cluster evaluation: colocation, KL divergence, hierarchical agglomeration
│   ├── vbgmm.py                   # Variational Bayes GMM implementation 
│   ├── helpers.py                 # General data handling and preprocessing helpers
│   ├── genericfuncs.py            # Utility functions
├── unsupervised_analysis.py   # Clustering pipeline: QIn loading, merging, superclustering, effect sizes
│   └── krnridgeclass.py           # Kernel ridge regression classifier
│
├── notebooks/       # Analysis notebooks (see table below)
├── figures/         # Output directory for all generated figures for the paper
└── results/         # Computed outputs: cross-validation results, QIn matrices, VBGMM results for all methods
```

To use `ppi.lib` in a notebook or script, add it to your path:

```python
import sys
sys.path.append("../ppi.lib/")
```

---

## Notebooks

The notebooks are used to produce the results of the paper. 

| Notebook | Description |
|---|---|
| `data_overview.ipynb` | Dataset exploration and summary statistics. Includes feature abbreviation tables and feature descriptions. |
| `evaltranspred.ipynb` | Evaluation of predictive performance using AUC, ROC, accuracy and Modified Shannon Capacity. Covers supervised models (SVC, MLP, GPC, RF) and trained GNN embeddings. |
| `aaphyche2rankmetrics.ipynb` | Feature importance ranking via Random Forest, GPC, Bayes Factor, McNemar p-value, feature-specific AUC and ACC (SVC). Applied to aggregated physicochemical features. |
| `clusterdata.ipynb` | Unsupervised clustering of interface features using GMM and VBGMM on both aggregated features and GNN embeddings. Includes elbow analysis for determining the number of clusters. |
| `unsupervised_analysis.ipynb` | Kruskal-Wallis effect size analysis across superclusters. Identifies which physicochemical features discriminate the discovered interface clusters. |
| `genfigures.ipynb` | Generates all publication figures from precomputed results. |

---

## Methods Summary

**Supervised analysis:** SVC, MLP, GPC and RF classifiers are trained on aggregated physicochemical features. A GNN (with different readout functions) is additionally trained on residue-level interface graphs, with the learned embeddings used as an alternative representation. Classification performance is evaluated using nested cross-validation.

**Feature importance:** Multiple complementary metrics (SHAP, MDI, permutation importance, KL divergence) are computed to rank features by their contribution to transmembrane vs. soluble interface discrimination.

**Unsupervised analysis:** GMM and VBGMM are applied to both feature representations to discover interface clusters without using transmembrane labels. Soft cluster assignments (QIn matrices) are hierarchically agglomerated into superclusters using symmetric KL divergence. Kruskal-Wallis effect sizes with Benjamini-Hochberg correction quantify feature discriminability across the discovered clusters.

---

## Input Data Format

The `source/` directory contains the following files:

| File | Description |
|---|---|
| `agg_final.csv` | Aggregated physicochemical feature matrix. One row per PPI, with `ExpID` as identifier. |
| `agg_trans.csv` | Transformed aggregated physicochemical feature matrix. |
| `GNN_readoutFunction_ALL_FOLDS.csv` | GNN embedding matrix (set2set pooling), one row per PPI. |
| `feature_names2_accronyms.csv` | Mapping of full feature names to short abbreviations used in figures. |

> **Note:** Raw structure files and MD simulation trajectories cannot be distributed due to size constraints. Precomputed feature matrices and QIn results are provided in `source/` and `results/` respectively.

---

## Results Directory

Pre-computed outputs of the clustering in `results/` include e.g.:

| File pattern | Description |
|---|---|
| `classical_agg_10_GMM_QIn.npy` | Soft cluster assignments (QIn) from GMM with 10 clusters on aggregated features |
| `classical_agg_10_vbgmm_QIn.npy` | QIn from VBGMM with 10 components on aggregated features |
| `classical_agg_34_vbgmm_QIn.npy` | QIn from VBGMM with 34 components on aggregated features |
| `GNN_s2s_10_GMM_QIn.npy` | QIn from GMM with 10 clusters on GNN (set2set) embeddings |
| `GNN_s2s_10_vbgmm_QIn.npy` | QIn from VBGMM with 10 components on GNN embeddings |

---

## Citation

> Manuscript in preparation.
