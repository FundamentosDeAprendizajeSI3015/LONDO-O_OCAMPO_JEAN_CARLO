"""
Supervised Learning Module
===========================
Trains and evaluates supervised classification models for network intrusion
detection. These models use labeled data (normal vs attack) to learn
decision boundaries.

Models implemented:
    - Decision Tree Classifier
    - Logistic Regression
    - Linear Regression (used as classifier with threshold)

Author: Jean Carlo Londoño Ocampo
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

FIGURE_DIR = "../reports/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Categorical features for one-hot encoding
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Features to drop (same as in the original project for consistency)
DROP_FEATURES = [
    "num_outbound_cmds",
    "num_root",
    "srv_serror_rate",
    "dst_host_srv_serror_rate",
    "srv_rerror_rate",
    "dst_host_srv_rerror_rate",
]

def prepare_supervised_data(train_df, test_df):
    """
    Prepare data for supervised learning.

    Unlike the unsupervised approach (train on normal only), supervised models
    need both classes during training.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set with 'label' column (values: 'normal' or attack types).
    test_df : pd.DataFrame
        Test set with same structure.

    Returns
    -------
    X_train, X_test : np.ndarray
        Scaled feature matrices.
    y_train, y_test : np.ndarray
        Binary labels (0 = normal, 1 = attack).
    feature_names : list
        Names of features after encoding.
    """
    # Convert labels to binary: normal=0, any attack=1
    y_train = (train_df["label"] != "normal").astype(int).values
    y_test = (test_df["label"] != "normal").astype(int).values

    # Drop target and metadata columns
    X_train = train_df.drop(columns=["label", "difficulty"], errors="ignore")
    X_test = test_df.drop(columns=["label", "difficulty"], errors="ignore")

    # Drop redundant features identified during data quality audit
    X_train = X_train.drop(columns=DROP_FEATURES, errors="ignore")
    X_test = X_test.drop(columns=DROP_FEATURES, errors="ignore")

    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS)

    # Align columns (test may have services not seen in training and vice versa)
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

    feature_names = X_train.columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def train_decision_tree(X_train, y_train, max_depth=10, random_state=42):
    """
    Train a Decision Tree classifier.

    Decision trees split data using feature thresholds that maximize
    information gain. Interpretable and handles non-linear relationships.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Binary training labels.
    max_depth : int
        Maximum tree depth (controls overfitting).

    Returns
    -------
    model : DecisionTreeClassifier
        Trained model.
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    print("[Decision Tree] Training complete.")
    return model

def train_logistic_regression(X_train, y_train, max_iter=1000, random_state=42):
    """
    Train a Logistic Regression classifier.

    Models the probability of a sample being an attack using the logistic
    (sigmoid) function. Linear decision boundary, works well when classes
    are approximately linearly separable.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Binary training labels.

    Returns
    -------
    model : LogisticRegression
        Trained model.
    """
    model = LogisticRegression(max_iter=max_iter, random_state=random_state, solver="lbfgs")
    model.fit(X_train, y_train)
    print("[Logistic Regression] Training complete.")
    return model

def train_linear_regression_classifier(X_train, y_train, threshold=0.5):
    """
    Train a Linear Regression model used as a binary classifier.

    Linear regression predicts continuous values. By applying a threshold
    (default 0.5), we convert it into a classifier. This is not ideal
    (outputs can exceed [0,1]) but serves as a comparison baseline.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Binary training labels (0/1).
    threshold : float
        Classification threshold.

    Returns
    -------
    model : LinearRegression
        Trained model.
    threshold : float
        Threshold used for converting regression output to class.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[Linear Regression] Training complete.")
    return model, threshold

def evaluate_supervised_model(model, X_test, y_test, model_name, threshold=None):
    """
    Evaluate a supervised model and return metrics.

    Parameters
    ----------
    model : estimator
        Trained sklearn model.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        True binary test labels.
    model_name : str
        Name for display and reporting.
    threshold : float, optional
        If provided, model.predict gives continuous values that are
        thresholded (used for Linear Regression).

    Returns
    -------
    metrics : dict
        Dictionary with accuracy, precision, recall, f1.
    y_pred : np.ndarray
        Predicted labels.
    """
    if threshold is not None:
        # Linear Regression: threshold continuous output
        y_pred_raw = model.predict(X_test)
        y_pred = (y_pred_raw >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }

    print(f"\n[{model_name}] Evaluation:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    return metrics, y_pred

def plot_confusion_matrices(results, y_test, prefix=""):
    """
    Plot confusion matrices for all models side by side.

    Parameters
    ----------
    results : list of tuples
        Each tuple: (model_name, y_pred).
    y_test : np.ndarray
        True labels.
    prefix : str
        Filename prefix to distinguish original vs relabeled.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, results):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"])
        ax.set_title(f"{name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    fname = f"{prefix}confusion_matrices.png" if prefix else "confusion_matrices.png"
    plt.savefig(os.path.join(FIGURE_DIR, fname), dpi=300)
    plt.close()
    print(f"Figure saved: {fname}")