"""
Model Comparison Module
========================
Compares supervised model performance when trained on original labels
versus corrected (re-evaluated) labels.

This addresses the hypothesis that ~30% of labels may be incorrect,
and correcting them should improve model reliability.

Author: Jean Carlo Londoño Ocampo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FIGURE_DIR = "../reports/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def build_comparison_table(original_metrics, relabeled_metrics):
    """
    Build a comparison DataFrame between original and relabeled model metrics.

    Parameters
    ----------
    original_metrics : list of dict
        Metrics from models trained on original labels.
    relabeled_metrics : list of dict
        Metrics from models trained on corrected labels.

    Returns
    -------
    comparison_df : pd.DataFrame
        Combined table showing both sets of results side by side.
    """
    df_orig = pd.DataFrame(original_metrics)
    df_orig["dataset"] = "Original Labels"

    df_relab = pd.DataFrame(relabeled_metrics)
    df_relab["dataset"] = "Relabeled"

    comparison_df = pd.concat([df_orig, df_relab], ignore_index=True)

    print("\n=== Model Comparison: Original vs Relabeled ===")
    print(comparison_df.to_string(index=False))

    return comparison_df

def plot_comparison(comparison_df):
    """
    Create a grouped bar chart comparing model metrics across datasets.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from build_comparison_table.
    """
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    models = comparison_df["model"].unique()

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    for ax, metric in zip(axes, metrics):
        x = np.arange(len(models))
        width = 0.35

        orig_vals = comparison_df[comparison_df["dataset"] == "Original Labels"][metric].values
        relab_vals = comparison_df[comparison_df["dataset"] == "Relabeled"][metric].values

        ax.bar(x - width / 2, orig_vals, width, label="Original", color="#4c72b0")
        ax.bar(x + width / 2, relab_vals, width, label="Relabeled", color="#dd8452")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, fontsize=8)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    plt.suptitle("Supervised Models: Original vs Relabeled Labels", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "model_comparison.png"), dpi=300)
    plt.close()
    print("Figure saved: model_comparison.png")