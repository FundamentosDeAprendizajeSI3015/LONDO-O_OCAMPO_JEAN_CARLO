"""
Visualization module for the Titanic ML project.

This module contains all plotting functions for exploratory data analysis (EDA),
model predictions, and performance metrics evaluation. It handles the creation
and saving of publication-quality visualizations.

Functions:
    set_plot_theme: Configure the default matplotlib/seaborn theme
    plot_eda: Generate exploratory data analysis visualizations
    plot_correlation: Create a correlation heatmap of features
    plot_lr_train_test: Compare train/test distributions for regression
    plot_lr_predictions: Visualize regression model predictions
    plot_lr_actual_vs_pred: Create actual vs predicted scatter plots
    plot_class_balance: Show class distribution before/after balancing
    plot_logreg_predictions: Visualize classification predictions
    plot_confusion_matrix: Display confusion matrix for classification
"""

import numpy as np
import matplotlib

# Use non-interactive backend for server/batch processing without display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from titanic_ml.io_utils import save_figure

def set_plot_theme():
    """
    Configure the default visual theme for all matplotlib/seaborn plots.
    
    Applies consistent styling across all visualizations including grid style,
    color palette, and font scaling for professional appearance.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

def plot_eda(df_raw, colors, output_dir):
    """
    Generate a comprehensive exploratory data analysis visualization (6-panel figure).
    
    Args:
        df_raw (pd.DataFrame): Raw dataset to analyze
        colors (dict): Color palette dictionary for consistent styling
        output_dir (Path): Directory path where the figure will be saved
    
    Displays:
        - Top-left: Survival distribution (binary outcome)
        - Top-center: Passenger class distribution
        - Top-right: Age distribution histogram
        - Bottom-left: Fare distribution histogram
        - Bottom-center: Survival by sex relationship
        - Bottom-right: Scatter plot of Fare vs Age colored by survival
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Titanic Dataset - Exploratory Data Analysis", fontsize=16, y=1.02)

    # Subplot 1: Count plot showing survival outcome distribution
    sns.countplot(
        x="Survived",
        data=df_raw,
        ax=axes[0, 0],
        palette=[colors["neg"], colors["pos"]],  # Red for not survived, green for survived
        hue="Survived",
        legend=False,
    )
    axes[0, 0].set_title("Survival Distribution")
    axes[0, 0].set_xticklabels(["Did not survive (0)", "Survived (1)"])

    # Subplot 2: Distribution of passenger classes (1st, 2nd, 3rd)
    sns.countplot(x="Pclass", data=df_raw, ax=axes[0, 1], palette="viridis", hue="Pclass", legend=False)
    axes[0, 1].set_title("Passenger Class Distribution")

    # Subplot 3: Histogram of passenger ages (missing values excluded)
    df_raw["Age"].dropna().hist(bins=30, ax=axes[0, 2], color=colors["train"], edgecolor="white", alpha=0.8)
    axes[0, 2].set_title("Age Distribution")
    axes[0, 2].set_xlabel("Age")

    # Subplot 4: Histogram of passenger ticket fares
    df_raw["Fare"].hist(bins=40, ax=axes[1, 0], color=colors["lasso"], edgecolor="white", alpha=0.8)
    axes[1, 0].set_title("Fare Distribution")
    axes[1, 0].set_xlabel("Fare")

    # Subplot 5: Survival rates by passenger sex (gender influence analysis)
    sns.countplot(x="Sex", hue="Survived", data=df_raw, ax=axes[1, 1], palette=[colors["neg"], colors["pos"]])
    axes[1, 1].set_title("Survival by Sex")
    axes[1, 1].legend(title="Survived", labels=["No", "Yes"])

    # Subplot 6: Scatter plot showing relationship between age, fare, and survival
    survived_mask = df_raw["Survived"] == 1  # Boolean mask for survived passengers
    # Plot non-survivors in red
    axes[1, 2].scatter(
        df_raw.loc[~survived_mask, "Age"],
        df_raw.loc[~survived_mask, "Fare"],
        c=colors["neg"],
        alpha=0.4,
        label="Did not survive",
        s=20,
    )
    # Plot survivors in green
    axes[1, 2].scatter(
        df_raw.loc[survived_mask, "Age"],
        df_raw.loc[survived_mask, "Fare"],
        c=colors["pos"],
        alpha=0.4,
        label="Survived",
        s=20,
    )
    axes[1, 2].set_title("Fare vs Age by Survival")
    axes[1, 2].set_xlabel("Age")
    axes[1, 2].set_ylabel("Fare")
    axes[1, 2].legend()

    # Save the figure and close to free memory
    out_path = save_figure(fig, output_dir, "01_eda.png")
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")

def plot_correlation(df_clean, output_dir):
    """
    Create a correlation heatmap showing relationships between all numeric features.
    
    Args:
        df_clean (pd.DataFrame): Preprocessed dataset with numeric features
        output_dir (Path): Directory path where the figure will be saved
    
    Note: Uses lower triangle only to avoid redundant correlation values.
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    # Compute Pearson correlation matrix for numeric columns only
    corr = df_clean.corr(numeric_only=True)
    # Create mask for upper triangle to avoid redundant information
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Display heatmap with correlation values
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    out_path = save_figure(fig, output_dir, "02_correlation.png")
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")

def plot_lr_train_test(y_train_lr, y_test_lr, colors, output_dir):
    """
    Compare fare distributions between training and test sets.
    
    Args:
        y_train_lr (np.ndarray): Fare values from training set
        y_test_lr (np.ndarray): Fare values from test set
        colors (dict): Color palette dictionary
        output_dir (Path): Directory path where the figure will be saved
    
    Creates two subplots:
        - Left: Sorted values to identify distribution differences
        - Right: Probability density histograms for comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left plot: sorted values to visualize distribution patterns
    axes[0].scatter(range(len(y_train_lr)), np.sort(y_train_lr), c=colors["train"], alpha=0.5, s=15, label="Train")
    axes[0].scatter(range(len(y_test_lr)), np.sort(y_test_lr), c=colors["test"], alpha=0.5, s=15, label="Test")
    axes[0].set_title("Sorted Fare Values: Train vs Test")
    axes[0].set_xlabel("Sorted Index")
    axes[0].set_ylabel("Fare")
    axes[0].legend()

    # Right plot: normalized histograms for density comparison
    axes[1].hist(y_train_lr, bins=30, color=colors["train"], alpha=0.6, label="Train", edgecolor="white", density=True)
    axes[1].hist(y_test_lr, bins=30, color=colors["test"], alpha=0.6, label="Test", edgecolor="white", density=True)
    axes[1].set_title("Fare Distribution: Train vs Test")
    axes[1].set_xlabel("Fare")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    out_path = save_figure(fig, output_dir, "03_lr_train_test.png")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def plot_lr_predictions(y_test_lr, y_pred_ridge, y_pred_lasso, r2_ridge, mae_ridge, r2_lasso, mae_lasso, colors, output_dir):
    """
    Visualize linear regression predictions for Ridge and Lasso models.
    
    Args:
        y_test_lr (np.ndarray): Actual test set fare values
        y_pred_ridge (np.ndarray): Ridge model predictions
        y_pred_lasso (np.ndarray): Lasso model predictions
        r2_ridge (float): R-squared score for Ridge model
        mae_ridge (float): Mean Absolute Error for Ridge model
        r2_lasso (float): R-squared score for Lasso model
        mae_lasso (float): Mean Absolute Error for Lasso model
        colors (dict): Color palette dictionary
        output_dir (Path): Directory path where the figure will be saved
    
    Creates side-by-side comparison of model predictions with performance metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Sort indices to visualize predictions in order of actual values
    sort_idx = np.argsort(y_test_lr)
    y_test_sorted = y_test_lr[sort_idx]
    y_ridge_sorted = y_pred_ridge[sort_idx]
    y_lasso_sorted = y_pred_lasso[sort_idx]

    # Left plot: Ridge Regression predictions
    axes[0].scatter(range(len(y_test_sorted)), y_test_sorted, c=colors["test"], alpha=0.5, s=20, label="Actual (Test)")
    axes[0].plot(range(len(y_ridge_sorted)), y_ridge_sorted, c=colors["ridge"], linewidth=2, label="Ridge Predicted")
    axes[0].set_title(f"Ridge Regression\nR2={r2_ridge:.4f} | MAE={mae_ridge:.4f}", fontsize=13)
    axes[0].set_xlabel("Sorted Sample Index")
    axes[0].set_ylabel("Fare")
    axes[0].legend()

    # Right plot: Lasso Regression predictions
    axes[1].scatter(range(len(y_test_sorted)), y_test_sorted, c=colors["test"], alpha=0.5, s=20, label="Actual (Test)")
    axes[1].plot(range(len(y_lasso_sorted)), y_lasso_sorted, c=colors["lasso"], linewidth=2, label="Lasso Predicted")
    axes[1].set_title(f"Lasso Regression\nR2={r2_lasso:.4f} | MAE={mae_lasso:.4f}", fontsize=13)
    axes[1].set_xlabel("Sorted Sample Index")
    axes[1].set_ylabel("Fare")
    axes[1].legend()

    out_path = save_figure(fig, output_dir, "04_lr_predictions.png")
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")


def plot_lr_actual_vs_pred(y_test_lr, y_pred_ridge, y_pred_lasso, colors, output_dir):
    """
    Create residual plots showing actual vs predicted values for both models.
    
    Args:
        y_test_lr (np.ndarray): Actual test set fare values
        y_pred_ridge (np.ndarray): Ridge model predictions
        y_pred_lasso (np.ndarray): Lasso model predictions
        colors (dict): Color palette dictionary
        output_dir (Path): Directory path where the figure will be saved
    
    Perfect predictions fall on the diagonal line (y=x).
    Points above/below line indicate over/under predictions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Loop through both models for side-by-side comparison
    for ax, y_pred, name, color in [
        (axes[0], y_pred_ridge, "Ridge", colors["ridge"]),
        (axes[1], y_pred_lasso, "Lasso", colors["lasso"]),
    ]:
        # Scatter plot of actual vs predicted values
        ax.scatter(y_test_lr, y_pred, c=color, alpha=0.5, s=20, edgecolors="white", linewidth=0.3)
        # Add diagonal reference line for perfect predictions
        lims = [min(y_test_lr.min(), y_pred.min()), max(y_test_lr.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", alpha=0.6, linewidth=1, label="Perfect prediction")
        ax.set_title(f"{name}: Actual vs Predicted", fontsize=13)
        ax.set_xlabel("Actual Fare")
        ax.set_ylabel("Predicted Fare")
        ax.legend()

    out_path = save_figure(fig, output_dir, "05_lr_actual_vs_pred.png")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def plot_class_balance(class_counts, balanced_counts, colors, output_dir):
    """
    Compare class distribution before and after handling class imbalance.
    
    Args:
        class_counts (list): Original class counts [negative, positive]
        balanced_counts (list): Balanced class counts [negative, positive] after undersampling
        colors (dict): Color palette dictionary
        output_dir (Path): Directory path where the figure will be saved
    
    Demonstrates the effect of undersampling on dataset composition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Original class distribution (imbalanced)
    axes[0].bar(["Class 0\n(Not survived)", "Class 1\n(Survived)"], [class_counts[0], class_counts[1]], color=[colors["neg"], colors["pos"]], edgecolor="white")
    axes[0].set_title("Before Balancing", fontsize=13)
    axes[0].set_ylabel("Count")
    # Add value labels on top of bars
    for i, v in enumerate([class_counts[0], class_counts[1]]):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # Right plot: Balanced class distribution (undersampled)
    axes[1].bar(["Class 0\n(Not survived)", "Class 1\n(Survived)"], [balanced_counts[0], balanced_counts[1]], color=[colors["neg"], colors["pos"]], edgecolor="white")
    axes[1].set_title("After Balancing (Undersampling)", fontsize=13)
    axes[1].set_ylabel("Count")
    # Add value labels on top of bars
    for i, v in enumerate([balanced_counts[0], balanced_counts[1]]):
        axes[1].text(i, v + 5, str(v), ha="center", fontweight="bold")

    plt.suptitle("Class Distribution: Before vs After Balancing", fontsize=14)
    out_path = save_figure(fig, output_dir, "06_class_balance.png")
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")

def plot_logreg_predictions(y_test_log, y_pred_log, y_prob_log, acc, f1, colors, output_dir):
    """
    Visualize logistic regression classification results and predicted probabilities.
    
    Args:
        y_test_log (np.ndarray): Actual test set binary labels (0 or 1)
        y_pred_log (np.ndarray): Predicted class labels from logistic regression
        y_prob_log (np.ndarray): Predicted survival probabilities [0, 1]
        acc (float): Classification accuracy score
        f1 (float): F1 score (harmonic mean of precision and recall)
        colors (dict): Color palette dictionary
        output_dir (Path): Directory path where the figure will be saved
    
    Creates two subplots:
        - Left: Probability distribution histogram with decision threshold at 0.5
        - Right: Scatter plot showing correct and misclassified predictions
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: Distribution of predicted probabilities separated by actual class
    axes[0].hist(y_prob_log[y_test_log == 0], bins=25, alpha=0.7, color=colors["neg"], label="Not Survived (Actual)", edgecolor="white")
    axes[0].hist(y_prob_log[y_test_log == 1], bins=25, alpha=0.7, color=colors["pos"], label="Survived (Actual)", edgecolor="white")
    # Add vertical line at the default decision threshold (0.5)
    axes[0].axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Threshold (0.5)")
    axes[0].set_title("Predicted Probability Distribution", fontsize=13)
    axes[0].set_xlabel("Predicted Probability of Survival")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Right plot: Show correct vs misclassified predictions
    x_axis = np.arange(len(y_test_log))
    correct = y_pred_log == y_test_log  # Boolean mask for correct predictions
    # Plot correct predictions in green circles
    axes[1].scatter(x_axis[correct], y_test_log[correct], c=colors["pos"], alpha=0.5, s=25, label="Correct", marker="o")
    # Plot misclassified predictions in red X's
    axes[1].scatter(x_axis[~correct], y_test_log[~correct], c=colors["neg"], alpha=0.7, s=40, label="Misclassified", marker="x")
    axes[1].set_title(f"Logistic Regression Predictions\nAccuracy={acc:.4f} | F1={f1:.4f}", fontsize=13)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Actual Class")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Not Survived", "Survived"])
    axes[1].legend()

    out_path = save_figure(fig, output_dir, "07_logreg_predictions.png")
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")

def plot_confusion_matrix(cm, tn, fp, fn, tp, output_dir):
    """
    Visualize the confusion matrix for binary classification model.
    
    Args:
        cm (np.ndarray): 2x2 confusion matrix array
        tn (int): True Negatives (correctly predicted negative)
        fp (int): False Positives (incorrectly predicted positive)
        fn (int): False Negatives (incorrectly predicted negative)
        tp (int): True Positives (correctly predicted positive)
        output_dir (Path): Directory path where the figure will be saved
    
    The confusion matrix shows:
        - Diagonal: Correct predictions (TN and TP)
        - Off-diagonal: Errors (FP and FN)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # Create heatmap of confusion matrix
    sns.heatmap(
        cm,
        annot=True,  # Annotate cells with values
        fmt="d",     # Format as integers
        cmap="Blues", # Color scheme
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
        ax=ax,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 18},
    )

    # Add metric labels (TN, FP, FN, TP) as text annotations
    labels = [[f"TN = {tn}", f"FP = {fp}"], [f"FN = {fn}", f"TP = {tp}"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.72, labels[i][j], ha="center", va="center", fontsize=11, color="gray", style="italic")

    ax.set_title("Confusion Matrix - Logistic Regression", fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    out_path = save_figure(fig, output_dir, "08_confusion_matrix.png")
    plt.close(fig)
    print(f"[SAVED] {out_path}")