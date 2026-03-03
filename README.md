# slc-characterization

This repository contain the data and code for Machine learning-based classification of protein-protein interfaces, distinguishing transmembrane from soluble interfaces using aggregated physicochemical features and GNN. Features are derived from experimentally validated PPIs, AlphaFold3-predicted structures refined through molecular dynamics simulations.

This repository contains all data which can be used for reproducing the results of the paper.
The paper link will follow.

---

## Installations 

```bash
conda create -n slc-env 
conda activate slc-env
pip install numpy pandas scikit-learn 
```

---

## Repository Structure

```
SLC-characterization/
├── source/                # Column name selectors and feature preprocessing configs
├── ppi.lib/               # helpers, eval metrics, for producing the results
├── notebooks/             # Analysis notebooks, where you can see how the functions can be used
├── figures/               # Output folder of the generated figures of the notebooks
└── results/               # Outputs of the calculations (CV, GNN, training results)
```

---

## Notebooks

The following list gives an overview of the jupyter notebooks. The order is the execution order of the paper.

| Notebook | Description |
|---|---|
| `data_overview.ipynb` | Dataset exploration and summary statistics also includes tables of abbreviations of the features and feature description|
| `evaltranspred.ipynb` | Notebook for evaluation of predicitve performance. As metrics we used AUC, ROC, ACC, Modified Shannon capacity. As input for the evaluation we used the supervised models (SVC, MLP, GPC and RF) and the trained GNN embeddings. |
| `aaphyche2rankmetrics.ipynb` | Feature importance ranking via RFC, GPC, Bayes Factor, McNemar p-value, feature specific AUC, ACC (from SVC). These metrics are calculated on the aggregated features. |
| `clusterdata.ipynb` | Unsupervised clustering of interface archetypes of aggregated features and GNN embeddings (the best performed embedding based on the supervised analysis). For clustering the GMM and VBGMM methods are used. You can also find the elbow analysis on where the input dimensions are defined in the notebook. |
| `effect_size_unsupervised.ipynb` | Effect size analysis across clusters based on Kurskal Wallis effect size.|
| `genfigures.ipynb` | Notebook which produces the Figures in the Paper |

---

**Note**: The raw data files cannot be proposed due to the limitation of this repository.

## Citation

> Manuscript in preparation.
