"""
Clustering Module
=================
Implements multiple unsupervised clustering algorithms for network traffic analysis.
Each algorithm groups data points without using labels, allowing us to evaluate
whether natural structure in the data aligns with known attack/normal categories.

Algorithms implemented:
    - K-Means
    - Fuzzy C-Means
    - Subtractive Clustering
    - DBSCAN
    - Agglomerative (Hierarchical) Clustering

Author: Jean Carlo Londoño Ocampo
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)

try:
    import skfuzzy as fuzz
except ImportError:
    fuzz = None

import matplotlib.pyplot as plt
import seaborn as sns
import os

FIGURE_DIR = "../reports/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# =============================================================================
# 1. K-Means Clustering
# =============================================================================
def run_kmeans(X, n_clusters=2, random_state=42):
    """
    Apply K-Means clustering to the dataset.

    K-Means partitions data into n_clusters groups by minimizing within-cluster
    variance (sum of squared distances to centroids). Fast and scalable but
    assumes spherical, equally-sized clusters.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    n_clusters : int
        Number of clusters to form.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each sample (-1 not used here).
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    print(f"[K-Means] Cluster sizes: {np.bincount(labels)}")
    return labels

# =============================================================================
# 2. Fuzzy C-Means Clustering
# =============================================================================
def run_fuzzy_cmeans(X, n_clusters=2, m=2, error=0.005, maxiter=1000):
    """
    Apply Fuzzy C-Means clustering.

    Unlike K-Means, each point has a membership degree (0-1) for every cluster.
    This is useful when boundaries between normal and attack traffic are not
    crisp — a connection might be partially anomalous.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    n_clusters : int
        Number of clusters.
    m : float
        Fuzziness coefficient (m=1 is hard clustering, m>1 is softer).
    error : float
        Convergence threshold.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    labels : np.ndarray
        Hard cluster assignment (argmax of membership matrix).
    membership : np.ndarray
        Membership matrix (n_clusters, n_samples) with degree values.
    """
    if fuzz is None:
        raise ImportError(
            "skfuzzy is required for Fuzzy C-Means clustering; install it with "
            "'pip install scikit-fuzzy'."
        )

    # skfuzzy expects data as (n_features, n_samples), so we transpose
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, c=n_clusters, m=m, error=error, maxiter=maxiter, seed=42
    )

    # u is membership matrix (n_clusters, n_samples)
    labels = np.argmax(u, axis=0)
    print(f"[Fuzzy C-Means] Cluster sizes: {np.bincount(labels)}")
    print(f"[Fuzzy C-Means] Fuzzy Partition Coefficient (FPC): {fpc:.4f}")
    return labels, u

# =============================================================================
# 3. Subtractive Clustering
# =============================================================================
def run_subtractive_clustering(X, ra=0.5, rb_ratio=1.25, accept_ratio=0.5, reject_ratio=0.15):
    """
    Apply Subtractive Clustering to estimate cluster centers.

    This density-based method automatically determines the number of clusters.
    Each data point is evaluated as a potential cluster center based on the
    density of surrounding points. Good for discovering the natural number
    of groups without specifying K upfront.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    ra : float
        Cluster radius — defines the neighborhood for density calculation.
        Smaller ra = more clusters.
    rb_ratio : float
        Multiplier for the rejection radius (rb = ra * rb_ratio).
    accept_ratio : float
        Threshold to accept a point as a cluster center.
    reject_ratio : float
        Threshold to reject a point as a cluster center.

    Returns
    -------
    labels : np.ndarray
        Cluster assignments based on nearest center.
    centers : np.ndarray
        Discovered cluster centers.
    """
    n_samples, n_features = X.shape
    rb = ra * rb_ratio

    # Step 1: Compute density (potential) for each point
    # Points in dense regions get higher potential
    potentials = np.zeros(n_samples)
    for i in range(n_samples):
        distances = np.sum((X - X[i]) ** 2, axis=1)
        potentials[i] = np.sum(np.exp(-distances / (ra / 2) ** 2))

    centers = []
    center_potentials = []
    potentials_copy = potentials.copy()
    first_potential = np.max(potentials_copy)

    # Step 2: Iteratively select centers and reduce surrounding potentials
    while True:
        idx = np.argmax(potentials_copy)
        current_potential = potentials_copy[idx]

        # Accept/reject logic based on potential ratio
        if current_potential > accept_ratio * first_potential:
            centers.append(X[idx])
            center_potentials.append(current_potential)
        elif current_potential < reject_ratio * first_potential:
            break
        else:
            # Check if center is far enough from existing centers
            if len(centers) > 0:
                dmin = min(np.sqrt(np.sum((X[idx] - c) ** 2)) for c in centers)
                if dmin / ra + current_potential / first_potential >= 1:
                    centers.append(X[idx])
                    center_potentials.append(current_potential)
                else:
                    potentials_copy[idx] = 0
                    continue
            else:
                centers.append(X[idx])
                center_potentials.append(current_potential)

        # Reduce potentials near the selected center
        distances = np.sum((X - X[idx]) ** 2, axis=1)
        potentials_copy -= current_potential * np.exp(-distances / (rb / 2) ** 2)
        potentials_copy = np.maximum(potentials_copy, 0)

    centers = np.array(centers)
    n_centers = len(centers)
    print(f"[Subtractive] Found {n_centers} cluster centers")

    # Step 3: Assign each point to the nearest center
    labels = np.argmin(
        np.array([np.sum((X - c) ** 2, axis=1) for c in centers]).T,
        axis=1,
    )

    print(f"[Subtractive] Cluster sizes: {np.bincount(labels)}")
    return labels, centers

# =============================================================================
# 4. DBSCAN Clustering
# =============================================================================
def run_dbscan(X, eps=1.5, min_samples=10):
    """
    Apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Groups together points that are closely packed and marks low-density points
    as noise (-1). Advantages: no need to specify number of clusters, naturally
    detects outliers (potential attacks).

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    eps : float
        Maximum distance between two samples to be considered neighbors.
    min_samples : int
        Minimum points required to form a dense region (core point).

    Returns
    -------
    labels : np.ndarray
        Cluster assignments. -1 indicates noise/outlier.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)
    print(f"[DBSCAN] Clusters found: {n_clusters}, Noise points: {n_noise}")

    return labels

# =============================================================================
# 5. Agglomerative (Hierarchical) Clustering
# =============================================================================
def run_agglomerative(X, n_clusters=2, linkage="ward"):
    """
    Apply Agglomerative Hierarchical Clustering.

    Builds clusters bottom-up: each point starts as its own cluster, then
    the two closest clusters are merged iteratively until n_clusters remain.
    'ward' linkage minimizes variance increase when merging.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Desired number of clusters.
    linkage : str
        Merge strategy: 'ward', 'complete', 'average', 'single'.

    Returns
    -------
    labels : np.ndarray
        Cluster assignments.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    print(f"[Agglomerative] Cluster sizes: {np.bincount(labels)}")
    return labels

# =============================================================================
# Evaluation Utilities
# =============================================================================
def evaluate_clustering(true_labels, predicted_labels, method_name, X=None):
    """
    Evaluate clustering quality using both internal and external metrics.

    External metrics (require true labels):
        - Adjusted Rand Index (ARI): Measures agreement between true and predicted.
          Range [-1, 1], 1 = perfect match.
        - Normalized Mutual Information (NMI): Measures shared information.
          Range [0, 1], 1 = perfect match.

    Internal metric (does not require true labels):
        - Silhouette Score: Measures how similar a point is to its own cluster
          vs other clusters. Range [-1, 1], higher = better separation.

    Parameters
    ----------
    true_labels : np.ndarray
        Ground truth labels (binary: 0=normal, 1=attack).
    predicted_labels : np.ndarray
        Cluster assignments from the algorithm.
    method_name : str
        Name of the clustering method (for display).
    X : np.ndarray, optional
        Feature matrix (needed for silhouette score).

    Returns
    -------
    metrics : dict
        Dictionary with ARI, NMI, and optionally Silhouette Score.
    """
    # Filter out noise points (DBSCAN labels = -1) for fair comparison
    mask = predicted_labels != -1
    true_filtered = true_labels[mask]
    pred_filtered = predicted_labels[mask]

    ari = adjusted_rand_score(true_filtered, pred_filtered)
    nmi = normalized_mutual_info_score(true_filtered, pred_filtered)

    metrics = {"method": method_name, "ARI": ari, "NMI": nmi}

    # Silhouette score requires at least 2 clusters with >1 sample each
    if X is not None and len(set(pred_filtered)) > 1:
        # Verify each cluster has enough samples
        unique_preds, pred_counts = np.unique(pred_filtered, return_counts=True)
        if all(c > 1 for c in pred_counts):
            sample_size = min(10000, len(pred_filtered))
            idx = np.random.RandomState(42).choice(len(pred_filtered), sample_size, replace=False)
            try:
                sil = silhouette_score(X[mask][idx], pred_filtered[idx])
                metrics["Silhouette"] = sil
            except ValueError:
                metrics["Silhouette"] = np.nan

    print(f"\n[{method_name}] Evaluation:")
    for k, v in metrics.items():
        if k != "method":
            print(f"  {k}: {v:.4f}")

    return metrics

def plot_clustering_comparison(results_df):
    """
    Create a grouped bar chart comparing all clustering methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: method, ARI, NMI, Silhouette.
    """
    metrics = [c for c in results_df.columns if c != "method"]
    x = np.arange(len(results_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            ax.bar(x + i * width, results_df[metric], width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["method"], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Clustering Algorithm Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "clustering_comparison.png"), dpi=300)
    plt.close()
    print("Figure saved: clustering_comparison.png")