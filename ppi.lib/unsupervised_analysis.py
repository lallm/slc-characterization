# =============================================================================
# unsupervised_analysis.py
#
# Unsupervised clustering analysis of protein-protein interfaces (PPIs).
# Covers:
#   1. Loading and cleaning soft cluster assignment matrices (QIn)
#   2. Merging undersized clusters into their nearest neighbour
#   3. Hierarchical agglomeration into superclusters
#   4. Statistical effect-size analysis per feature per cluster
#   5. Visualisation: dendrograms, bar charts, heatmaps
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, levene, bartlett, chi2_contingency, kruskal, f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap  # for heatmap colourmap
from scipy.cluster import hierarchy                    # dendrogram plotting
from scipy.cluster.hierarchy import fcluster          # cluster extraction from linkage


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_qin_clean(qin_path):
    """
    Loads QIn (allocation probabilities), removes clusters that contain no data 
    points (unused), and remaps labels to a clean consecutive integer range.
    """
    # Load the raw QIn matrix from disk (shape: n_samples x n_clusters)
    Q = np.load(qin_path)

    # Determine which cluster columns are actually used (i.e. the argmax of at
    # least one sample points to that column). VBGMM often produces empty columns.
    used = np.unique(Q.argmax(axis=1))

    # Keep only the populated cluster columns to avoid indexing into empty ones
    Q = Q[:, used]

    # Convert soft assignments to hard labels via argmax, then re-index to 0-based
    # consecutive integers 
    _, labels = np.unique(Q.argmax(axis=1), return_inverse=True)

    return Q, labels


def merge_small_clusters(Q, labels, min_size=10):
    """
    Identifies clusters smaller than min_size and merges them into the 
    most similar neighbouring cluster based on the mean QIn probability.

    Clusters below min_size are artefacts of VBGMM over-segmentation and 
    are not statistically reliable for downstream analysis.
    """
    # Work on a copy so the original labels array is not modified in place
    labels_clean = labels.copy()

    while True:
        # Count how many samples belong to each cluster
        counts = pd.Series(labels_clean).value_counts()

        # Collect the IDs of all clusters that fall below the size threshold
        tiny = counts[counts < min_size].index.tolist()

        # Exit the loop once all clusters are large enough
        if not tiny:
            break

        # Process one tiny cluster per iteration (the first one found)
        t = tiny[0]

        # Boolean mask identifying samples that currently belong to cluster t
        mask = labels_clean == t

        # All other cluster IDs that are candidates for absorption
        other = [c for c in np.unique(labels_clean) if c != t]

        # Select the most similar neighbour: the cluster whose mean allocation
        # probability Q[:, c] is highest for the samples in cluster t.
        # This is equivalent to finding the cluster these samples are
        # "almost assigned to" according to the soft model.
        nearest = max(other, key=lambda c: Q[mask][:, c].mean())

        # Reassign all samples of cluster t to the nearest cluster
        labels_clean[mask] = nearest

        # Re-index labels to consecutive integers after each merge to avoid
        # stale references in the next iteration
        _, labels_clean = np.unique(labels_clean, return_inverse=True)

    # Final re-indexing pass to guarantee no gaps remain in the label range
    _, labels_clean = np.unique(labels_clean, return_inverse=True)

    return labels_clean


# ─────────────────────────────────────────────────────────────────────────────
# 2. HIERARCHICAL SUPERCLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_superclusters(Q, labels_clean, evms, min_size=10):
    """
    Performs hierarchical agglomeration on the cleaned cluster labels to group
    them into 4 'superclusters', then ensures no supercluster is smaller than
    min_size by merging any tiny one into the largest.

    The agglomeration uses the symmetric KL divergence (Jeffreys distance)
    between cluster-level QIn distributions as the inter-cluster distance metric.
    """
    # Build a linkage matrix by agglomerating the fine-grained clusters.
    # QIn2symkl computes symmetric KL divergence between cluster QIn distributions.
    # davgquant with qantval=1.0 uses the maximum quantile for aggregation
    _, linkage = evms.agglomerate_multi_allocprobs(
        [Q],
        [labels_clean.astype(int)],
        QI_metric=evms.QIn2symkl,
        aggsamples=lambda v: evms.davgquant(v, qantval=1.0, isqntlower=True)
    )

    # Cut the dendrogram at a level that yields exactly 4 flat superclusters,
    # then propagate the supercluster label of each fine cluster to each sample.
    # fcluster returns a label per fine cluster; indexing by labels_clean maps
    # each sample to its supercluster.
    sc = fcluster(linkage, t=4, criterion="maxclust")[labels_clean]

    # Post-processing: if any supercluster is smaller than min_size, merge it
    # into the largest supercluster to maintain statistical reliability.
    while True:
        counts = pd.Series(sc).value_counts()

        # Stop once all superclusters meet the minimum size requirement
        if (counts >= min_size).all():
            break

        # Identify the smallest and largest superclusters by sample count
        small = counts.idxmin()
        large = counts.idxmax()

        # Absorb the smallest supercluster into the largest
        sc[sc == small] = large

    # Re-index supercluster IDs to consecutive integers
    _, sc = np.unique(sc, return_inverse=True)

    return sc


def build_supercluster_df(exp_path, agg_df, sc):
    """
    Merges the supercluster assignments with the original physicochemical 
    features (agg_df) using the Experiment ID as the join key.
    """
    # Load experiment metadata; standardise the binary label column name to 'label'
    # regardless of whether the source file uses 'y_true' or 'Transmembrane'
    exp = pd.read_csv(exp_path).rename(
        columns={"y_true": "label", "Transmembrane": "label"}
    )

    # Build the base DataFrame with one row per sample, carrying the supercluster label
    df = pd.DataFrame({
        "ExpID":        exp["ExpID"].values,
        "label":        exp["label"].astype(int).values,   # 0 = soluble, 1 = transmembrane
        "supercluster": sc
    })

    # Exclude columns that would either conflict with or duplicate existing columns
    # when merging the physicochemical features from agg_df
    drop_cols = {"cluster_assignment", "supercluster", "label"}
    merge_cols = [c for c in agg_df.columns if c not in drop_cols]

    # Ensure 'ExpID' is always present as the merge key
    if "ExpID" not in merge_cols:
        merge_cols = ["ExpID"] + merge_cols

    # Left merge: keep all rows from df (every sample with a supercluster)
    # and attach corresponding physicochemical features from agg_df.
    # drop_duplicates ensures one feature row per unique interface.
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
    Calculates the discriminative power of each physicochemical feature
    across the clusters defined by cluster_col.

    Test selection:
      - Binary features (only 0/1 values): Chi-squared test + Cramer's V effect size.
      - Continuous features:
          * Default: Kruskal-Wallis H-test + epsilon-squared (e2) effect size.
          * use_anova=True: one-way ANOVA + eta-squared effect size.

    Multiple-testing correction is applied via Benjamini-Hochberg FDR.
    """

    def bh_correct(pvals):
        """
        Applies Benjamini-Hochberg (BH) False Discovery Rate correction to a
        list of raw p-values and returns the adjusted p-values.

        BH guarantees that the expected proportion of false discoveries among
        all rejected null hypotheses is controlled at level 0.05.
        Formula: p_adj[i] = p[i] * n / rank[i], clipped to [0, 1].
        """
        pvals = np.array(pvals)
        n = len(pvals)

        # Rank p-values from smallest (rank 1) to largest (rank n)
        order = np.argsort(pvals)
        ranked = np.empty(n)
        ranked[order] = np.arange(1, n + 1)

        # BH adjustment: multiply each p-value by (n / its rank), then cap at 1
        return np.minimum(1, pvals * n / ranked)

    results = []

    # Determine the sorted cluster IDs for consistent group ordering
    cluster_order = sorted(df[cluster_col].unique())

    # Select only numeric feature columns; skip metadata and cluster columns
    feat_cols = [
        c for c in df.columns
        if c not in ["ExpID", "label", "supercluster"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for feat in feat_cols:
        # Drop rows where either the cluster label or the feature value is NaN
        clean = df[[cluster_col, feat]].dropna()

        # Skip constant features — they have no discriminative power
        if clean[feat].nunique() < 2:
            continue

        # Detect binary features: those whose unique values are a subset of {0, 1}
        is_binary = set(clean[feat].unique()).issubset({0, 1, 0.0, 1.0})

        try:
            if is_binary:
                # ── Binary feature branch ─────────────────────────────────────
                # Build a contingency table: rows = clusters, columns = 0/1 values
                ct = pd.crosstab(clean[cluster_col], clean[feat])

                # Need at least two value columns to test for association
                if ct.shape[1] < 2:
                    continue

                # Chi-squared test of independence between cluster and binary feature
                chi2, p, _, _ = chi2_contingency(ct)

                # Cramer's V: normalised Chi-squared, ranges [0, 1]
                # V = sqrt(chi2 / (N * (min(rows, cols) - 1)))
                eff = np.sqrt(chi2 / (ct.sum().sum() * (min(ct.shape) - 1)))
                test = "Chi2"

            else:
                # ── Continuous feature branch ─────────────────────────────────
                # Build one array per cluster, retaining only clusters with more then 3 samples
                groups = [
                    clean.loc[clean[cluster_col] == c, feat].values
                    for c in cluster_order
                    if len(clean.loc[clean[cluster_col] == c]) >= 3
                ]

                # Need at least two groups to perform any group comparison
                if len(groups) < 2:
                    continue

                if use_anova:
                    # Parametric one-way ANOVA (assumes normality + homoscedasticity)
                    stat, p = f_oneway(*groups)
                    test = "ANOVA"
                else:
                    # Non-parametric Kruskal-Wallis H-test (rank-based, no normality assumption)
                    stat, p = kruskal(*groups)
                    test = "KW"

                # Epsilon-squared (e2): effect size for Kruskal-Wallis
                # Formula: e2 = H / ((n^2 - 1) / (n + 1))
                # Where H = KW test statistic, n = total number of observations.
                # e2 ranges [0, 1]; 0 = no effect, 1 = perfect separation.
                n = sum(len(g) for g in groups)
                eff = max(0, stat / ((n ** 2 - 1) / (n + 1)))

            # Append the result for this feature
            results.append({
                "feature": feat,
                "effect":  eff,
                "p_value": p,
                "test":    test
            })

        except Exception:
            # Skip features that raise unexpected errors (e.g. degenerate distributions)
            continue

    # Assemble the results DataFrame
    res = pd.DataFrame(results)

    if len(res):
        # Apply BH correction across all tested features simultaneously
        res["p_adj"] = bh_correct(res["p_value"])

        # Flag features whose adjusted p-value passes the 5% significance threshold
        res["sig"] = res["p_adj"] < 0.05

    # Return features ranked from highest to lowest discriminative effect size
    return res.sort_values("effect", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_dendrograms(Q, labels_clean, evms, title_prefix="Model"):
    """
    Plots two side-by-side dendrograms to compare the hierarchical relationship
    among fine-grained clusters under two different distance metrics:
      - Colocation : default evms metric based on co-assignment probabilities.
      - Jeffreys   : symmetric KL divergence (QIn2symkl) between cluster distributions.
    """
    # Collect unique cluster IDs and build human-readable labels with sample counts
    unique = np.unique(labels_clean)
    label_names = [f"C{c} (n={(labels_clean == c).sum()})" for c in unique]

    # Compute linkage matrix using the default Colocation metric
    _, linkage_coloc = evms.agglomerate_multi_allocprobs(
        [Q], [labels_clean.astype(int)],
        aggsamples=lambda v: evms.davgquant(v, qantval=1.0, isqntlower=True)
    )

    # Compute linkage matrix using the symmetric KL (Jeffreys) divergence metric
    _, linkage_jeff = evms.agglomerate_multi_allocprobs(
        [Q], [labels_clean.astype(int)],
        QI_metric=evms.QIn2symkl,
        aggsamples=lambda v: evms.davgquant(v, qantval=1.0, isqntlower=True)
    )

    # Create a figure with two subplots side by side.
    # Height scales with the number of clusters to keep leaf labels readable.
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(unique) * 0.4)))

    # Render each dendrogram with horizontal (left) orientation for readability
    for ax, lnk, name in zip(axes, [linkage_coloc, linkage_jeff], ["Colocation", "Jeffreys"]):
        hierarchy.dendrogram(
            lnk,
            labels=np.array(label_names),
            orientation="left",
            leaf_font_size=10,
            ax=ax
        )
        ax.set_title(f"{title_prefix}\n{name}", fontweight="bold")

        # Remove top and right spines for a cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_final_clustering_effects(eff_mat, abbrev_csv_path):
    """
    Generates a grouped bar chart comparing the discriminative effect sizes of 
    the top physicochemical features across three clustering models:
    AGG-GMM, AGG-VBGMM, and AGG-VBGMM34.

    The union of the top-7 features from each model is shown, sorted by
    AGG-GMM effect size. Feature names are replaced with short abbreviations
    for axis readability.
    """
    # Load the abbreviation lookup table and build a dict for fast mapping
    abbrev_df  = pd.read_csv(abbrev_csv_path)
    abbrev_map = dict(zip(abbrev_df['features'], abbrev_df['features_abbreviation']))

    # Remove rows corresponding to the ground-truth label feature —
    # we are interested in unsupervised discriminability, not supervised signal
    eff_mat_plot = eff_mat.drop(
        index=[i for i in eff_mat.index
               if 'label' in i.lower() or 'transmembrane' in i.lower()]
    )

    # The three models to compare and their assigned bar colours
    methods = ['AGG-GMM', 'AGG-VBGMM', 'AGG-VBGMM34']
    colors  = {'AGG-GMM': '#4C72B0', 'AGG-VBGMM': '#DD8452', 'AGG-VBGMM34': '#55A868'}

    # Collect the top-7 feature names for each model, then form their union.
    # dict.fromkeys preserves insertion order while deduplicating entries.
    top_gmm   = eff_mat_plot['AGG-GMM'].nlargest(7).index
    top_vbg   = eff_mat_plot['AGG-VBGMM'].nlargest(7).index
    top_vbg34 = eff_mat_plot['AGG-VBGMM34'].nlargest(7).index
    all_feats = list(dict.fromkeys(list(top_gmm) + list(top_vbg) + list(top_vbg34)))

    # Subset the effect-size matrix to the selected features and chosen models
    plot_df = eff_mat_plot.loc[all_feats, methods].copy()

    # Replace full feature names with their short abbreviations on the x-axis;
    # if a feature has no abbreviation entry, fall back to the original name
    plot_df.index = [abbrev_map.get(f, f) for f in plot_df.index]

    # Sort bars by AGG-GMM effect size (largest first) for visual clarity
    plot_df = plot_df.sort_values('AGG-GMM', ascending=False)

    # Draw the grouped bar chart
    ax = plot_df.plot.bar(
        figsize=(9.6, 4.8),
        color=[colors[m] for m in methods],
        width=0.75
    )
    ax.set_ylabel('Effect Size (e2)', fontsize=16)
    ax.set_title('Feature Discriminability across Clustering Methods', fontsize=18)

    # Rotate x-axis labels 90 degrees to prevent overlap with long names
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()  # uncomment when running interactively


def plot_heatmap(effects_dict, top_n=64):
    """
    Generates a high-resolution heatmap of feature effect sizes across all
    clustering models. Statistically significant features (BH-adjusted p < 0.05)
    are marked with an asterisk (*) in the top-right corner of their cell.
    """
    # Gather the superset of all feature names across all models
    all_feats = sorted(
        set().union(*[set(df['feature']) for df in effects_dict.values()])
    )
    models = list(effects_dict.keys())

    # Initialise effect-size and significance matrices with NaN / False
    eff_mat = pd.DataFrame(np.nan,  index=all_feats, columns=models)
    sig_mat = pd.DataFrame(False,   index=all_feats, columns=models)

    # Populate the matrices from each model's results DataFrame
    for name, res in effects_dict.items():
        for _, r in res.iterrows():
            eff_mat.loc[r['feature'], name] = r['effect']
            sig_mat.loc[r['feature'], name] = r.get('sig', False)

    # Select the top_n features by their mean effect size across all models.
    # This summarises overall discriminability rather than per-model ranking.
    top_feats_idx = eff_mat.mean(axis=1).nlargest(top_n).index
    eff_top = eff_mat.loc[top_feats_idx].astype(float)

    # Reset index on sig_mat so iloc-based positional access is aligned with eff_top
    sig_top = sig_mat.loc[top_feats_idx].reset_index(drop=True)

    # Clean up feature names for display: replace underscores with spaces and
    # rename the ground-truth label column to the more descriptive 'Transmembrane'
    eff_top.index = [
        f.replace('_', ' ').replace('label', 'Transmembrane')
        for f in eff_top.index
    ]

    # Custom sequential colormap: white (effect=0) -> orange -> red (effect=1)
    cmap = LinearSegmentedColormap.from_list(
        'eff_size', ['#f7f7f7', '#fc8d59', '#d73027']
    )

    # Figure height scales with the number of feature rows so labels are not cramped
    fig, ax = plt.subplots(figsize=(10, max(8, len(eff_top) * 0.4)))

    sns.heatmap(
        eff_top,
        ax=ax,
        cmap=cmap,
        annot=True,       # print numeric effect size in each cell
        fmt='.2f',        # two decimal places
        linewidths=0.5,   # thin gridlines between cells for readability
        linecolor='#dddddd'
    )

    # Overprint an asterisk in the upper-right of each cell where the
    # BH-adjusted p-value is below 0.05 (statistically significant effect)
    for i in range(len(eff_top)):
        for j in range(len(eff_top.columns)):
            if sig_top.iloc[i, j]:
                ax.text(
                    j + 0.85, i + 0.25, '*',
                    color='black', fontsize=12,
                    fontweight='bold', ha='center'
                )

    ax.set_title(
        'Feature Discriminability (KW, BH * p < 0.05)',
        fontweight='bold', fontsize=14
    )
    plt.tight_layout()
    plt.show()
