"""
Extended Pipeline - Main Module
================================
Orchestrates the complete extended analysis:
    1. Load and prepare data
    2. Run unsupervised clustering algorithms
    3. Re-evaluate labels using clustering consensus
    4. Train supervised models on original labels
    5. Train supervised models on corrected labels
    6. Compare results

Author: Jean Carlo Londoño Ocampo
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from clustering import (
    run_kmeans,
    run_fuzzy_cmeans,
    run_subtractive_clustering,
    run_dbscan,
    run_agglomerative,
    evaluate_clustering,
    plot_clustering_comparison,
)
from relabeling import (
    align_cluster_labels,
    majority_vote_relabeling,
    generate_relabeled_dataset,
)
from supervised import (
    prepare_supervised_data,
    train_decision_tree,
    train_logistic_regression,
    train_linear_regression_classifier,
    evaluate_supervised_model,
    plot_confusion_matrices,
)
from comparison import build_comparison_table, plot_comparison

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_PATH = "../data/KDDTrain_.txt"
TEST_PATH = "../data/KDDTest_.txt"
RANDOM_STATE = 42
SAMPLE_SIZE_CLUSTERING = 8000  # Subsample for expensive algorithms
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DROP_FEATURES = [
    "num_outbound_cmds", "num_root", "srv_serror_rate",
    "dst_host_srv_serror_rate", "srv_rerror_rate", "dst_host_srv_rerror_rate",
]

def prepare_for_clustering(df, sample_size=None):
    """
    Prepare data for clustering: encode, scale, and optionally subsample.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix.
    binary_labels : np.ndarray
        Binary labels (0=normal, 1=attack) for evaluation.
    sampled_df : pd.DataFrame
        The (possibly subsampled) DataFrame.
    """
    # Convert labels to binary
    binary_labels = (df["label"] != "normal").astype(int).values

    # Drop non-feature columns
    X = df.drop(columns=["label", "difficulty"], errors="ignore")
    X = X.drop(columns=DROP_FEATURES, errors="ignore")

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Subsample if requested (for computational efficiency)
    if sample_size and sample_size < len(X_scaled):
        np.random.seed(RANDOM_STATE)
        idx = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_scaled = X_scaled[idx]
        binary_labels = binary_labels[idx]
        df = df.iloc[idx].reset_index(drop=True)

    return X_scaled, binary_labels, df

def main():
    print("=" * 60)
    print("  EXTENDED ANALYSIS PIPELINE")
    print("=" * 60)

    # ── 1. Load Data ───────────────────────────────────────────────────────
    print("\n[Step 1] Loading data...")
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # Merge and balance (same strategy as original project)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    normal_df = full_df[full_df["label"] == "normal"]
    attack_df = full_df[full_df["label"] != "normal"]
    sample_size = min(len(normal_df), len(attack_df))

    sampled_normal = normal_df.sample(n=sample_size, random_state=RANDOM_STATE)
    sampled_attack = attack_df.sample(n=sample_size, random_state=RANDOM_STATE)
    balanced_df = pd.concat([sampled_normal, sampled_attack])
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Split into train/test
    train_data, test_data = train_test_split(
        balanced_df, test_size=0.3, random_state=RANDOM_STATE, stratify=balanced_df["label"].apply(
            lambda x: 0 if x == "normal" else 1
        )
    )

    print(f"  Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(f"  Train label distribution:\n{train_data['label'].value_counts().to_string()}")

    # ── 2. Unsupervised Clustering ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Step 2] Running unsupervised clustering algorithms...")
    print("=" * 60)

    # Prepare data for clustering (use training set)
    X_clust, y_binary, clust_df = prepare_for_clustering(
        train_data, sample_size=SAMPLE_SIZE_CLUSTERING
    )

    clustering_results = {}
    evaluation_results = []

    # 2a. K-Means
    print("\n--- K-Means ---")
    km_labels = run_kmeans(X_clust, n_clusters=2)
    km_aligned = align_cluster_labels(y_binary, km_labels)
    clustering_results["K-Means"] = km_aligned
    evaluation_results.append(evaluate_clustering(y_binary, km_aligned, "K-Means", X_clust))

    # 2b. Fuzzy C-Means
    print("\n--- Fuzzy C-Means ---")
    fcm_labels, fcm_membership = run_fuzzy_cmeans(X_clust, n_clusters=2)
    fcm_aligned = align_cluster_labels(y_binary, fcm_labels)
    clustering_results["Fuzzy C-Means"] = fcm_aligned
    evaluation_results.append(evaluate_clustering(y_binary, fcm_aligned, "Fuzzy C-Means", X_clust))

    # 2c. Subtractive Clustering (use smaller sample due to O(n^2) complexity)
    print("\n--- Subtractive Clustering ---")
    sub_sample_size = min(5000, len(X_clust))
    np.random.seed(RANDOM_STATE)
    sub_idx = np.random.choice(len(X_clust), sub_sample_size, replace=False)
    sub_labels_small, sub_centers = run_subtractive_clustering(X_clust[sub_idx], ra=0.5)

    # Map subtractive results back: assign full dataset to nearest center
    from scipy.spatial.distance import cdist
    dists = cdist(X_clust, sub_centers, metric="euclidean")
    sub_labels_full = np.argmin(dists, axis=1)

    # Reduce to 2 clusters if subtractive found more (merge smallest)
    unique, counts = np.unique(sub_labels_full, return_counts=True)
    if len(unique) > 2:
        # Keep 2 largest clusters, merge rest into the nearest large one
        sorted_idx = np.argsort(-counts)
        top2 = set(unique[sorted_idx[:2]])
        for cl in unique:
            if cl not in top2:
                # Assign to nearest of top2 centers
                top2_centers = sub_centers[sorted_idx[:2]]
                nearest = sorted_idx[np.argmin(
                    [np.linalg.norm(sub_centers[cl] - c) for c in top2_centers]
                )]
                sub_labels_full[sub_labels_full == cl] = unique[nearest]
        # Renumber to 0,1
        mapping = {v: i for i, v in enumerate(sorted(set(sub_labels_full)))}
        sub_labels_full = np.array([mapping[l] for l in sub_labels_full])

    sub_aligned = align_cluster_labels(y_binary, sub_labels_full)
    clustering_results["Subtractive"] = sub_aligned
    evaluation_results.append(evaluate_clustering(y_binary, sub_aligned, "Subtractive", X_clust))

    # 2d. DBSCAN
    print("\n--- DBSCAN ---")
    dbscan_labels = run_dbscan(X_clust, eps=2.0, min_samples=10)
    dbscan_aligned = align_cluster_labels(y_binary, dbscan_labels)
    clustering_results["DBSCAN"] = dbscan_aligned
    evaluation_results.append(evaluate_clustering(y_binary, dbscan_aligned, "DBSCAN", X_clust))

    # 2e. Agglomerative Clustering (memory-intensive, use smaller subsample)
    print("\n--- Agglomerative ---")
    agg_sample_size = min(5000, len(X_clust))
    np.random.seed(RANDOM_STATE)
    agg_idx = np.random.choice(len(X_clust), agg_sample_size, replace=False)
    agg_labels_small = run_agglomerative(X_clust[agg_idx], n_clusters=2)
    # Extend to full sample via nearest-centroid assignment
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_clust[agg_idx], agg_labels_small)
    agg_labels = knn.predict(X_clust)
    agg_aligned = align_cluster_labels(y_binary, agg_labels)
    clustering_results["Agglomerative"] = agg_aligned
    evaluation_results.append(evaluate_clustering(y_binary, agg_aligned, "Agglomerative", X_clust))

    # Plot clustering comparison
    eval_df = pd.DataFrame(evaluation_results)
    plot_clustering_comparison(eval_df)
    print("\nClustering evaluation summary:")
    print(eval_df.to_string(index=False))

    # ── 3. Re-evaluate Labels ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Step 3] Re-evaluating labels using clustering consensus...")
    print("=" * 60)

    # Run fast clustering algorithms on the full training set for relabeling
    X_full, y_full_binary, _ = prepare_for_clustering(train_data)

    full_clustering = {}

    # K-Means (fast)
    km_full = run_kmeans(X_full, n_clusters=2)
    full_clustering["K-Means"] = align_cluster_labels(y_full_binary, km_full)

    # Fuzzy C-Means (moderate)
    fcm_full, _ = run_fuzzy_cmeans(X_full, n_clusters=2)
    full_clustering["Fuzzy C-Means"] = align_cluster_labels(y_full_binary, fcm_full)

    # Run KMeans with different K to add diversity
    km3 = run_kmeans(X_full, n_clusters=3)
    # Map 3 clusters to binary via alignment
    km3_aligned = align_cluster_labels(y_full_binary, km3)
    full_clustering["K-Means-3"] = km3_aligned

    # Majority vote: flip only when ALL methods disagree (strict threshold)
    new_labels, flip_mask, relabel_stats = majority_vote_relabeling(
        y_full_binary, full_clustering, threshold=0.6
    )

    # Create relabeled training set
    train_data_relabeled = generate_relabeled_dataset(train_data, new_labels)

    # ── 4. Supervised Models on ORIGINAL Labels ────────────────────────────
    print("\n" + "=" * 60)
    print("[Step 4] Training supervised models on ORIGINAL labels...")
    print("=" * 60)

    X_train_s, X_test_s, y_train_s, y_test_s, feat_names = prepare_supervised_data(
        train_data, test_data
    )

    original_metrics = []
    original_preds = []

    # Decision Tree
    dt_model = train_decision_tree(X_train_s, y_train_s)
    dt_metrics, dt_pred = evaluate_supervised_model(dt_model, X_test_s, y_test_s, "Decision Tree")
    original_metrics.append(dt_metrics)
    original_preds.append(("Decision Tree", dt_pred))

    # Logistic Regression
    lr_model = train_logistic_regression(X_train_s, y_train_s)
    lr_metrics, lr_pred = evaluate_supervised_model(lr_model, X_test_s, y_test_s, "Logistic Reg.")
    original_metrics.append(lr_metrics)
    original_preds.append(("Logistic Reg.", lr_pred))

    # Linear Regression (as classifier)
    lin_model, lin_thresh = train_linear_regression_classifier(X_train_s, y_train_s)
    lin_metrics, lin_pred = evaluate_supervised_model(
        lin_model, X_test_s, y_test_s, "Linear Reg.", threshold=lin_thresh
    )
    original_metrics.append(lin_metrics)
    original_preds.append(("Linear Reg.", lin_pred))

    plot_confusion_matrices(original_preds, y_test_s, prefix="original_")

    # ── 5. Supervised Models on RELABELED Data ─────────────────────────────
    print("\n" + "=" * 60)
    print("[Step 5] Training supervised models on RELABELED data...")
    print("=" * 60)

    X_train_r, X_test_r, y_train_r, y_test_r, _ = prepare_supervised_data(
        train_data_relabeled, test_data
    )

    relabeled_metrics = []
    relabeled_preds = []

    # Decision Tree
    dt_model_r = train_decision_tree(X_train_r, y_train_r)
    dt_m_r, dt_p_r = evaluate_supervised_model(dt_model_r, X_test_r, y_test_r, "Decision Tree")
    relabeled_metrics.append(dt_m_r)
    relabeled_preds.append(("Decision Tree", dt_p_r))

    # Logistic Regression
    lr_model_r = train_logistic_regression(X_train_r, y_train_r)
    lr_m_r, lr_p_r = evaluate_supervised_model(lr_model_r, X_test_r, y_test_r, "Logistic Reg.")
    relabeled_metrics.append(lr_m_r)
    relabeled_preds.append(("Logistic Reg.", lr_p_r))

    # Linear Regression
    lin_model_r, lin_thresh_r = train_linear_regression_classifier(X_train_r, y_train_r)
    lin_m_r, lin_p_r = evaluate_supervised_model(
        lin_model_r, X_test_r, y_test_r, "Linear Reg.", threshold=lin_thresh_r
    )
    relabeled_metrics.append(lin_m_r)
    relabeled_preds.append(("Linear Reg.", lin_p_r))

    plot_confusion_matrices(relabeled_preds, y_test_r, prefix="relabeled_")

    # ── 6. Comparison ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Step 6] Comparing original vs relabeled models...")
    print("=" * 60)

    comparison_df = build_comparison_table(original_metrics, relabeled_metrics)
    plot_comparison(comparison_df)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "clustering_eval": eval_df,
        "relabel_stats": relabel_stats,
        "original_metrics": original_metrics,
        "relabeled_metrics": relabeled_metrics,
        "comparison": comparison_df,
    }

if __name__ == "__main__":
    results = main()