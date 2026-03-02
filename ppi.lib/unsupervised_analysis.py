import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, levene, bartlett, chi2_contingency, kruskal, f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_qin_clean(qin_path):
    """
    Loads QIn (allocation probabilities), removes clusters that contain no data 
    points (unused), and remaps labels to a clean consecutive integer range.
    """
    Q = np.load(qin_path)

    # Find unique cluster indices that are actually assigned to at least one sample
    used = np.unique(Q.argmax(axis=1))
    Q = Q[:, used]

    # Remap original cluster IDs (e.g., 0, 5, 12) to consecutive IDs (e.g., 0, 1, 2)
    _, labels = np.unique(Q.argmax(axis=1), return_inverse=True)

    return Q, labels


def merge_small_clusters(Q, labels, min_size=10):
    """
    Identifies clusters smaller than min_size and merges them into the 
    most similar neighboring cluster based on the mean QIn probability.
    """
    labels_clean = labels.copy()

    while True:
        counts = pd.Series(labels_clean).value_counts()
        tiny = counts[counts < min_size].index.tolist()
        if not tiny:
            break

        # Process the first tiny cluster found
        t = tiny[0]
        mask = labels_clean == t
        other = [c for c in np.unique(labels_clean) if c != t]

        # Find the nearest cluster by maximizing the average allocation probability (Q)
        nearest = max(other, key=lambda c: Q[mask][:, c].mean())
        labels_clean[mask] = nearest

    # Final re-indexing to ensure no gaps in the integer labels
    _, labels_clean = np.unique(labels_clean, return_inverse=True)
    return labels_clean


# ─────────────────────────────────────────────────────────────────────────────
# 2. HIERARCHICAL SUPERCLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_superclusters(Q, labels_clean, evms, min_size=10):
    """
    Performs hierarchical clustering on the cleaned labels to group them into 
    4 'superclusters'. Also ensures no supercluster is below the min_size.
    """
    # Generate linkage matrix using the specified metric (Jeffreys/SymKL)
    _, linkage = evms.agglomerate_multi_allocprobs(
        [Q],
        [labels_clean.astype(int)],
        QI_metric=evms.QIn2symkl,
        aggsamples=lambda v: evms.davgquant(v, qantval=1.0, isqntlower=True)
    )

    # Cut the tree to produce exactly 4 superclusters
    sc = fcluster(linkage, t=4, criterion="maxclust")[labels_clean]

    # Post-merge loop: If a supercluster is too small, merge it into the largest one
    while True:
        counts = pd.Series(sc).value_counts()
        if (counts >= min_size).all():
            break
        small = counts.idxmin()
        large = counts.idxmax()
        sc[sc == small] = large

    return sc


def build_supercluster_df(exp_path, agg_df, sc):
    """
    Merges the supercluster assignments with the original physicochemical 
    features (agg_df) using the Experiment ID as the key.
    """
    # Load experiment metadata and standardize label column names
    exp = pd.read_csv(exp_path).rename(
        columns={"y_true": "label", "Transmembrane": "label"}
    )

    # Create base dataframe with IDs and the new supercluster labels
    df = pd.DataFrame({
        "ExpID": exp["ExpID"].values,
        "label": exp["label"].astype(int).values,
        "supercluster": sc
    })

    # Filter out columns we don't want to duplicate during the merge
    drop_cols = {"cluster_assignment", "supercluster", "label"}
    merge_cols = [c for c in agg_df.columns if c not in drop_cols]

    if "ExpID" not in merge_cols:
        merge_cols = ["ExpID"] + merge_cols

    # Left merge ensures we keep all samples that have a supercluster assignment
    return df.merge(
        agg_df[merge_cols].drop_duplicates("ExpID"),
        on="ExpID",
        how="left"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTICAL ANALYSIS (EFFECT SIZES)
# ─────────────────────────────────────────────────────────────────────────────

def compute_effects(df, cluster_col, use_anova=False):
    """
    Calculates the discriminative power of each feature across clusters.
    Uses Chi-squared for binary features and Kruskal-Wallis/ANOVA for continuous.
    Includes Benjamini-Hochberg (BH) p-value correction.
    """
    
    def bh_correct(pvals):
        """Internal helper for Benjamini-Hochberg False Discovery Rate correction."""
        pvals = np.array(pvals)
        n = len(pvals)
        order = np.argsort(pvals)
        ranked = np.empty(n)
        ranked[order] = np.arange(1, n + 1)
        return np.minimum(1, pvals * n / ranked)

    results = []
    cluster_order = sorted(df[cluster_col].unique())

    # Identify numeric columns only
    feat_cols = [
        c for c in df.columns
        if c not in ["ExpID", "label", "supercluster"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for feat in feat_cols:
        clean = df[[cluster_col, feat]].dropna()
        if clean[feat].nunique() < 2:
            continue

        # Check if feature is binary (e.g., 0 or 1)
        is_binary = set(clean[feat].unique()).issubset({0,1,0.0,1.0})

        try:
            if is_binary:
                # Use Chi-squared for categorical/binary data
                ct = pd.crosstab(clean[cluster_col], clean[feat])
                if ct.shape[1] < 2: continue
                chi2, p, _, _ = chi2_contingency(ct)
                # Effect size: Cramer's V
                eff = np.sqrt(chi2 / (ct.sum().sum() * (min(ct.shape)-1)))
                test = "Chi2"
            else:
                # Prepare groups for continuous data tests
                groups = [
                    clean.loc[clean[cluster_col]==c, feat].values
                    for c in cluster_order
                    if len(clean.loc[clean[cluster_col]==c]) >= 3
                ]
                if len(groups) < 2: continue

                if use_anova:
                    stat, p = f_oneway(*groups)
                    test = "ANOVA"
                else:
                    stat, p = kruskal(*groups)
                    test = "KW"

                # Effect size: Epsilon-squared (ε²) for non-parametric data
                n = sum(len(g) for g in groups)
                k = len(groups)
                eff = max(0, (stat - (k-1)) / (n-k))

            results.append({
                "feature": feat, "effect": eff, "p_value": p, "test": test
            })
        except:
            continue

    # Finalize results table with adjusted p-values
    res = pd.DataFrame(results)
    if len(res):
        res["p_adj"] = bh_correct(res["p_value"])
        res["sig"] = res["p_adj"] < 0.05

    return res.sort_values("effect", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_dendrograms(Q, labels_clean, evms, title_prefix="Model"):
    """
    Plots two dendrograms side-by-side to compare cluster relationships 
    using Colocation metrics and Jeffreys divergence.
    """
    unique = np.unique(labels_clean)
    label_names = [f"C{c} (n={(labels_clean==c).sum()})" for c in unique]

    # Compute linkages for both metrics
    _, linkage_coloc = evms.agglomerate_multi_allocprobs(
        [Q], [labels_clean.astype(int)],
        aggsamples=lambda v: evms.davgquant(v, qantval=1.0, isqntlower=True)
    )

    _, linkage_jeff = evms.agglomerate_multi_allocprobs(
        [Q], [labels_clean.astype(int)],
        QI_metric=evms.QIn2symkl,
        aggsamples=lambda v: evms.davgquant(v, qantval=1.0, isqntlower=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(unique)*0.4)))

    for ax, lnk, name in zip(axes, [linkage_coloc, linkage_jeff], ["Colocation", "Jeffreys"]):
        hierarchy.dendrogram(
            lnk, labels=np.array(label_names), orientation="left",
            leaf_font_size=10, ax=ax
        )
        ax.set_title(f"{title_prefix}\n{name}", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()



def plot_final_clustering_effects(eff_mat, abbrev_csv_path):
    """
    Generates a grouped bar chart comparing the top 7 features of 
    three specific clustering models (AGG-GMM, VBGMM, VBGMM34).
    """
    # Load and map feature abbreviations for cleaner X-axis labels
    abbrev_df = pd.read_csv(abbrev_csv_path)
    abbrev_map = dict(zip(abbrev_df['features'], abbrev_df['features_abbreviation']))

    # Drop non-relevant features
    eff_mat_plot = eff_mat.drop(
        index=[i for i in eff_mat.index if 'label' in i.lower() or 'transmembrane' in i.lower()]
    )
    
    methods = ['AGG-GMM', 'AGG-VBGMM', 'AGG-VBGMM34']
    colors = {'AGG-GMM': '#4C72B0', 'AGG-VBGMM': '#DD8452', 'AGG-VBGMM34': '#55A868'}

    # Get the union of the top 7 features from all 3 models
    top_gmm = eff_mat_plot['AGG-GMM'].nlargest(7).index
    top_vbg = eff_mat_plot['AGG-VBGMM'].nlargest(7).index
    top_vbg34 = eff_mat_plot['AGG-VBGMM34'].nlargest(7).index
    all_feats = list(dict.fromkeys(list(top_gmm) + list(top_vbg) + list(top_vbg34)))

    # Sort and label
    plot_df = eff_mat_plot.loc[all_feats, methods].copy()
    plot_df.index = [abbrev_map.get(f, f) for f in plot_df.index]
    plot_df = plot_df.sort_values('AGG-GMM', ascending=False)

    # Plot configuration
    ax = plot_df.plot.bar(figsize=(9.6, 4.8), color=[colors[m] for m in methods], width=0.75)
    ax.set_ylabel('Effect Size (ε²)', fontsize=16)
    ax.set_title('Feature Discriminability across Clustering Methods', fontsize=18)
    plt.xticks(rotation=90)
    plt.tight_layout()
    #plt.show()


def plot_heatmap(effects_dict, top_n=64):
    """
    Generates a high-resolution heatmap for all models, highlighting 
    statistically significant features with an asterisk (*).
    """
    # Align results from different dictionaries into a single Matrix
    all_feats = sorted(set().union(*[set(df['feature']) for df in effects_dict.values()]))
    models = list(effects_dict.keys())
    
    eff_mat = pd.DataFrame(np.nan, index=all_feats, columns=models)
    sig_mat = pd.DataFrame(False,  index=all_feats, columns=models)
    
    for name, res in effects_dict.items():
        for _, r in res.iterrows():
            eff_mat.loc[r['feature'], name] = r['effect']
            sig_mat.loc[r['feature'], name] = r.get('sig', False)

    # Filtering for top N features and cleaning names
    top_feats_idx = eff_mat.mean(axis=1).nlargest(top_n).index
    eff_top = eff_mat.loc[top_feats_idx].astype(float)
    sig_top = sig_mat.loc[top_feats_idx].reset_index(drop=True)
    eff_top.index = [f.replace('_', ' ').replace('label', 'Transmembrane') for f in eff_top.index]

    # custom colormap
    cmap = LinearSegmentedColormap.from_list('eff_size', ['#f7f7f7', '#fc8d59', '#d73027'])

    fig, ax = plt.subplots(figsize=(10, max(8, len(eff_top) * 0.4)))
    sns.heatmap(eff_top, ax=ax, cmap=cmap, annot=True, fmt='.2f', linewidths=0.5, linecolor='#dddddd')

    # Overprint asterisks for significant features
    for i in range(len(eff_top)):
        for j in range(len(eff_top.columns)):
            if sig_top.iloc[i, j]:
                ax.text(j + 0.85, i + 0.25, '*', color='black', fontsize=12, fontweight='bold', ha='center')

    ax.set_title('Feature Discriminability (KW, BH * p < 0.05)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()