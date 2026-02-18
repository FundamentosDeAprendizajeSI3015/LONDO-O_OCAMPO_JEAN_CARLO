"""
Visualization Module
Provides graphical analysis for data exploration and model evaluation.
Saves figures automatically inside the project repository.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# Directory where figures will be stored
FIGURE_DIR = "../reports/figures"

# Create directory if it does not exist
os.makedirs(FIGURE_DIR, exist_ok=True)


def save_figure(filename):
    """
    Save current matplotlib figure to reports directory.
    """
    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {path}")


def plot_label_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df)
    plt.title("Label Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure("label_distribution.png")
    plt.show()


def plot_feature_distribution(df, column):
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], bins=50, kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    save_figure(f"distribution_{column}.png")
    plt.show()


def plot_anomaly_score_distribution(scores):
    plt.figure(figsize=(6,4))
    sns.histplot(scores, bins=50, kde=True)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.tight_layout()
    save_figure("anomaly_score_distribution.png")
    plt.show()


def plot_score_by_class(scores, true_labels):
    plt.figure(figsize=(6,4))
    
    sns.histplot(scores[true_labels == 1], bins=50, color='blue', label='Normal', kde=True)
    sns.histplot(scores[true_labels == -1], bins=50, color='red', label='Attack', kde=True)
    
    plt.legend()
    plt.title("Anomaly Score Distribution by Class")
    plt.xlabel("Anomaly Score")
    plt.tight_layout()
    save_figure("anomaly_score_by_class.png")
    plt.show()


def plot_pca_projection(X_scaled, true_labels):
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
    save_figure("pca_projection.png")
    plt.show()