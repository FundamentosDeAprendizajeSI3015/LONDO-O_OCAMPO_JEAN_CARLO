"""
Visualization Module
Provides graphical analysis for data exploration and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

def plot_label_distribution(df):
    """
    Plot distribution of normal vs attack labels.
    """
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df)
    plt.title("Label Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_feature_distribution(df, column):
    """
    Plot histogram of a numerical feature.
    """
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], bins=50, kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()


def plot_anomaly_score_distribution(scores):
    """
    Plot distribution of anomaly scores.
    """
    plt.figure(figsize=(6,4))
    sns.histplot(scores, bins=50, kde=True)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.tight_layout()
    plt.show()


def plot_score_by_class(scores, true_labels):
    """
    Compare anomaly score distribution for normal vs attack samples.
    """
    plt.figure(figsize=(6,4))
    
    sns.histplot(scores[true_labels == 1], bins=50, color='blue', label='Normal', kde=True)
    sns.histplot(scores[true_labels == -1], bins=50, color='red', label='Attack', kde=True)
    
    plt.legend()
    plt.title("Anomaly Score Distribution by Class")
    plt.xlabel("Anomaly Score")
    plt.tight_layout()
    plt.show()


def plot_pca_projection(X_scaled, true_labels):
    """
    Project high-dimensional data into 2D using PCA.
    Useful for visualizing separation between normal and attack traffic.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(6,6))
    plt.scatter(
        X_pca[true_labels == 1, 0],
        X_pca[true_labels == 1, 1],
        alpha=0.5,
        label="Normal"
    )
    plt.scatter(
        X_pca[true_labels == -1, 0],
        X_pca[true_labels == -1, 1],
        alpha=0.5,
        label="Attack"
    )
    
    plt.legend()
    plt.title("PCA Projection (2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.show()