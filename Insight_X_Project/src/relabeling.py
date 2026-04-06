"""
Label Re-evaluation Module
===========================
Uses consensus from multiple clustering algorithms to identify and correct
potentially mislabeled samples in the dataset.

Strategy:
    1. Run multiple clustering algorithms (each produces binary labels).
    2. Align cluster labels to true labels (handle label permutation).
    3. Use majority voting: if most algorithms disagree with the original
       label, flag it as potentially mislabeled.
    4. Generate a corrected label set for supervised training.

Author: Jean Carlo Londoño Ocampo
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def align_cluster_labels(true_labels, cluster_labels):
    """
    Align cluster labels to true labels using the Hungarian algorithm.

    Clustering algorithms assign arbitrary label numbers (e.g., cluster 0
    might correspond to "attack" or "normal"). This function finds the
    optimal mapping that maximizes agreement with true labels.

    Parameters
    ----------
    true_labels : np.ndarray
        Ground truth binary labels (0 = normal, 1 = attack).
    cluster_labels : np.ndarray
        Raw cluster assignments from an algorithm.

    Returns
    -------
    aligned_labels : np.ndarray
        Cluster labels remapped to match true label semantics.
    """
    # Handle noise labels from DBSCAN (-1) by treating them as a separate cluster
    unique_clusters = np.unique(cluster_labels)
    unique_true = np.unique(true_labels)

    # Build cost matrix: rows = cluster IDs, cols = true label IDs
    # We want to maximize matches, so we use negative counts as cost
    n_clusters = len(unique_clusters)
    n_classes = len(unique_true)
    size = max(n_clusters, n_classes)
    cost_matrix = np.zeros((size, size))

    for i, cl in enumerate(unique_clusters):
        for j, tl in enumerate(unique_true):
            cost_matrix[i, j] = -np.sum((cluster_labels == cl) & (true_labels == tl))

    # Hungarian algorithm finds optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping from cluster label to true label
    mapping = {}
    for i, cl in enumerate(unique_clusters):
        if i < len(col_ind) and col_ind[i] < len(unique_true):
            mapping[cl] = unique_true[col_ind[i]]
        else:
            mapping[cl] = unique_true[0]  # Default fallback

    aligned_labels = np.array([mapping.get(l, 0) for l in cluster_labels])
    return aligned_labels

def majority_vote_relabeling(true_labels, clustering_results, threshold=0.5):
    """
    Re-evaluate labels using majority voting across clustering algorithms.

    For each sample, if more than `threshold` fraction of algorithms disagree
    with the original label, the label is flipped.

    Parameters
    ----------
    true_labels : np.ndarray
        Original binary labels (0 = normal, 1 = attack).
    clustering_results : dict
        Dictionary mapping algorithm name to its aligned cluster labels.
        Example: {"K-Means": array([0, 1, 1, ...]), "DBSCAN": array([0, 0, 1, ...])}
    threshold : float
        Fraction of algorithms that must disagree to flip a label.
        0.5 means simple majority.

    Returns
    -------
    new_labels : np.ndarray
        Corrected labels after majority voting.
    flip_mask : np.ndarray
        Boolean mask indicating which samples were relabeled.
    stats : dict
        Statistics about the relabeling process.
    """
    n_samples = len(true_labels)
    n_methods = len(clustering_results)

    # Count how many algorithms assign each sample to the "opposite" class
    disagreement_count = np.zeros(n_samples)

    for method_name, aligned_labels in clustering_results.items():
        # Count disagreements with original labels
        disagreement_count += (aligned_labels != true_labels).astype(int)

    # Flip labels where majority of algorithms disagree
    disagreement_ratio = disagreement_count / n_methods
    flip_mask = disagreement_ratio > threshold

    new_labels = true_labels.copy()
    new_labels[flip_mask] = 1 - new_labels[flip_mask]  # Flip 0->1 or 1->0

    # Compute statistics
    n_flipped = np.sum(flip_mask)
    pct_flipped = 100 * n_flipped / n_samples

    # Breakdown: how many normal->attack and attack->normal
    normal_to_attack = np.sum(flip_mask & (true_labels == 0))
    attack_to_normal = np.sum(flip_mask & (true_labels == 1))

    stats = {
        "total_samples": n_samples,
        "total_flipped": int(n_flipped),
        "pct_flipped": round(pct_flipped, 2),
        "normal_to_attack": int(normal_to_attack),
        "attack_to_normal": int(attack_to_normal),
    }

    print("\n=== Label Re-evaluation Results ===")
    print(f"  Total samples: {n_samples}")
    print(f"  Labels flipped: {n_flipped} ({pct_flipped:.2f}%)")
    print(f"  Normal -> Attack: {normal_to_attack}")
    print(f"  Attack -> Normal: {attack_to_normal}")

    return new_labels, flip_mask, stats

def generate_relabeled_dataset(df, new_labels):
    """
    Create a new DataFrame with corrected labels.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset with 'label' column.
    new_labels : np.ndarray
        Corrected binary labels (0 = normal, 1 = attack).

    Returns
    -------
    relabeled_df : pd.DataFrame
        Copy of original DataFrame with updated 'label' column.
    """
    relabeled_df = df.copy()
    relabeled_df["label"] = np.where(new_labels == 0, "normal", "attack")
    return relabeled_df