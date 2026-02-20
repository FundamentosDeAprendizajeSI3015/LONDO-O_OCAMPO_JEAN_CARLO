"""
Model training and evaluation module for the Titanic ML project.

This module implements two main prediction pipelines:

1. Linear Regression: Predicts passenger fare prices using Ridge and Lasso models
   with hyperparameter tuning via RandomizedSearchCV cross-validation.

2. Logistic Regression: Predicts survival outcomes using various regularization
   techniques with class balancing and stratified cross-validation.

Functions:
    run_linear_regression: Train and evaluate regression models for fare prediction
    run_logistic_regression: Train and evaluate classification models for survival prediction
"""

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from titanic_ml.plots import (
    plot_class_balance,
    plot_confusion_matrix,
    plot_logreg_predictions,
    plot_lr_actual_vs_pred,
    plot_lr_predictions,
    plot_lr_train_test,
)

def run_linear_regression(df_clean, colors, output_dir, random_state=42):
    """
    Train and evaluate linear regression models (Ridge and Lasso) for fare prediction.
    
    This function performs the following steps:
    1. Removes fare outliers (99th percentile)
    2. Splits data into training and test sets
    3. Performs hyperparameter tuning using RandomizedSearchCV with 5-fold CV
    4. Evaluates both Ridge and Lasso models on test set
    5. Generates visualizations and returns performance metrics
    
    Args:
        df_clean (pd.DataFrame): Preprocessed dataset with all features
        colors (dict): Color palette dictionary for visualization consistency
        output_dir (Path): Directory where visualization plots will be saved
        random_state (int): Random seed for reproducibility. Default: 42
    
    Returns:
        dict: Dictionary containing R² and MAE scores for both models, plus optimal hyperparameters
    """
    # Remove extreme outliers using 99th percentile to improve model robustness
    fare_99 = df_clean["Fare"].quantile(0.99)
    df_lr = df_clean[df_clean["Fare"] <= fare_99].copy()
    print(f"\nRecords after outlier removal: {len(df_lr)} (removed {len(df_clean) - len(df_lr)} outliers)")

    # Define target variable and features (exclude target and irrelevant columns)
    target_lr = "Fare"
    features_lr = [c for c in df_lr.columns if c not in [target_lr, "Survived"]]
    X_lr = df_lr[features_lr].values
    y_lr = df_lr[target_lr].values

    print(f"Features used: {features_lr}")
    print(f"X shape: {X_lr.shape}, y shape: {y_lr.shape}")

    # Stratified split: 80% train, 20% test
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=random_state)
    print(f"\nTrain size: {X_train_lr.shape[0]}, Test size: {X_test_lr.shape[0]}")

    # Visualize training vs test distribution
    plot_lr_train_test(y_train_lr, y_test_lr, colors, output_dir)

    # Create pipelines with feature scaling and regularized regression models
    # StandardScaler normalizes features to have mean=0 and std=1 for better convergence
    pipeline_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    pipeline_lasso = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(max_iter=10000))])

    # Define hyperparameter grid using log-uniform distributions (good for alpha/regularization)
    param_dist_ridge = {"ridge__alpha": loguniform(1e-3, 1e3)}
    param_dist_lasso = {"lasso__alpha": loguniform(1e-4, 1e2)}
    # KFold split without stratification (appropriate for regression)
    cv_lr = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Hyperparameter tuning for Ridge Regression using RandomizedSearchCV
    # Samples 100 random combinations of alpha parameters and evaluates using 5-fold CV
    print("\n--- Tuning Ridge ---")
    rs_ridge = RandomizedSearchCV(
        estimator=pipeline_ridge,
        param_distributions=param_dist_ridge,
        n_iter=100,  # Number of random combinations to try
        scoring="r2",  # Optimize for R² score (coefficient of determination)
        cv=cv_lr,  # Use KFold cross-validation
        random_state=random_state,
        n_jobs=-1,  # Use all available CPU cores for parallel processing
        verbose=0,  # Suppress detailed iteration output
    )
    rs_ridge.fit(X_train_lr, y_train_lr)

    # Hyperparameter tuning for Lasso Regression
    # Lasso (L1 regularization) performs feature selection by shrinking some coefficients to zero
    print("--- Tuning Lasso ---")
    rs_lasso = RandomizedSearchCV(
        estimator=pipeline_lasso,
        param_distributions=param_dist_lasso,
        n_iter=100,
        scoring="r2",
        cv=cv_lr,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    rs_lasso.fit(X_train_lr, y_train_lr)

    print(f"\n{'='*50}")
    print(f"RIDGE - Best parameters : {rs_ridge.best_params_}")
    print(f"RIDGE - Best CV R2      : {rs_ridge.best_score_:.4f}")
    print(f"{'='*50}")
    print(f"LASSO - Best parameters : {rs_lasso.best_params_}")
    print(f"LASSO - Best CV R2      : {rs_lasso.best_score_:.4f}")
    print(f"{'='*50}")

    # Generate predictions on test set and evaluate using multiple metrics
    y_pred_ridge = rs_ridge.predict(X_test_lr)
    r2_ridge = r2_score(y_test_lr, y_pred_ridge)  # R² score: 1 = perfect fit, 0 = baseline model
    mae_ridge = mean_absolute_error(y_test_lr, y_pred_ridge)  # MAE: average absolute prediction error

    y_pred_lasso = rs_lasso.predict(X_test_lr)
    r2_lasso = r2_score(y_test_lr, y_pred_lasso)
    mae_lasso = mean_absolute_error(y_test_lr, y_pred_lasso)

    print(f"\n{'='*50}")
    print("TEST SET METRICS")
    print(f"{'='*50}")
    print(f"{'Model':<10} {'R2':>10} {'MAE':>12}")
    print(f"{'-'*32}")
    print(f"{'Ridge':<10} {r2_ridge:>10.4f} {mae_ridge:>12.4f}")
    print(f"{'Lasso':<10} {r2_lasso:>10.4f} {mae_lasso:>12.4f}")
    print(f"{'='*50}")

    # Generate prediction visualizations
    plot_lr_predictions(y_test_lr, y_pred_ridge, y_pred_lasso, r2_ridge, mae_ridge, r2_lasso, mae_lasso, colors, output_dir)
    plot_lr_actual_vs_pred(y_test_lr, y_pred_ridge, y_pred_lasso, colors, output_dir)

    # Return dictionary with key metrics and hyperparameters for reporting
    return {
        "r2_ridge": r2_ridge,
        "mae_ridge": mae_ridge,
        "r2_lasso": r2_lasso,
        "mae_lasso": mae_lasso,
        "ridge_alpha": rs_ridge.best_params_["ridge__alpha"],
        "lasso_alpha": rs_lasso.best_params_["lasso__alpha"],
    }

def run_logistic_regression(df_clean, colors, output_dir, random_state=42):
    """
    Train and evaluate logistic regression models for survival prediction with class balancing.
    
    This function performs the following steps:
    1. Checks class imbalance in survival outcomes
    2. Applies random undersampling to balance classes
    3. Trains logistic regression with various regularization options
    4. Performs hyperparameter tuning using StratifiedKFold cross-validation
    5. Evaluates classification metrics (accuracy, F1-score, confusion matrix)
    6. Generates visualizations and returns performance metrics
    
    Args:
        df_clean (pd.DataFrame): Preprocessed dataset with all features
        colors (dict): Color palette dictionary for visualization consistency
        output_dir (Path): Directory where visualization plots will be saved
        random_state (int): Random seed for reproducibility. Default: 42
    
    Returns:
        dict: Dictionary containing accuracy, F1-score, confusion matrix elements,
              minority class count, and other classification metrics
    """
    # Define target and features for binary classification
    target_log = "Survived"
    features_log = [c for c in df_clean.columns if c != target_log]
    X_log = df_clean[features_log].values
    y_log = df_clean[target_log].values

    print(f"\nFeatures used: {features_log}")
    print(f"X shape: {X_log.shape}, y shape: {y_log.shape}")

    # Analyze class distribution (imbalance analysis)
    class_counts = pd.Series(y_log).value_counts()
    print("\nOriginal class distribution:")
    print(f"  Class 0 (Did not survive): {class_counts[0]} ({class_counts[0]/len(y_log)*100:.1f}%)")
    print(f"  Class 1 (Survived)       : {class_counts[1]} ({class_counts[1]/len(y_log)*100:.1f}%)")

    # Apply random undersampling to handle class imbalance
    # This technique reduces the majority class to match minority class size
    # Trade-off: May lose information but improves class balance for training
    print("\n--- Balancing the dataset (random undersampling of majority class) ---")

    df_balanced = df_clean.copy()
    class_0 = df_balanced[df_balanced[target_log] == 0]
    class_1 = df_balanced[df_balanced[target_log] == 1]
    minority_count = min(len(class_0), len(class_1))  # Size of smaller class
    majority_label = 0 if len(class_0) > len(class_1) else 1  # Identify majority class

    if len(class_0) != len(class_1):
        # Undersample the majority class to match minority class size
        if majority_label == 0:
            class_0_under = class_0.sample(n=minority_count, random_state=random_state)
            # Combine balanced classes and shuffle to avoid ordering bias
            df_balanced = pd.concat([class_0_under, class_1], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
        else:
            class_1_under = class_1.sample(n=minority_count, random_state=random_state)
            df_balanced = pd.concat([class_0, class_1_under], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

        X_log_bal = df_balanced[features_log].values
        y_log_bal = df_balanced[target_log].values
        print(f"Balanced dataset size: {len(df_balanced)}")
    else:
        # No balancing needed if classes already have equal size
        X_log_bal = X_log
        y_log_bal = y_log
        print("Dataset is already balanced.")

    balanced_counts = pd.Series(y_log_bal).value_counts()
    print("Balanced class distribution:")
    print(f"  Class 0: {balanced_counts[0]} ({balanced_counts[0]/len(y_log_bal)*100:.1f}%)")
    print(f"  Class 1: {balanced_counts[1]} ({balanced_counts[1]/len(y_log_bal)*100:.1f}%)")

    # Visualize the effect of class balancing
    plot_class_balance(class_counts, balanced_counts, colors, output_dir)

    # Stratified split ensures both train and test sets have similar class distribution
    # Important for classification to avoid biased evaluation
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X_log_bal,
        y_log_bal,
        test_size=0.2,  # Reserve 20% of balanced data for testing
        random_state=random_state,
        stratify=y_log_bal,  # Maintain class distribution in both sets
    )
    print(f"\nTrain size: {X_train_log.shape[0]}, Test size: {X_test_log.shape[0]}")
    print(f"Train class dist: {dict(zip(*np.unique(y_train_log, return_counts=True)))}") 
    print(f"Test  class dist: {dict(zip(*np.unique(y_test_log, return_counts=True)))}")

    # Build logistic regression pipeline with feature scaling
    pipeline_logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000, random_state=random_state)),
    ])

    # Define hyperparameter grid for extensive search
    # C: Inverse regularization strength (smaller C = stronger regularization)
    # penalty: L1 (Lasso) or L2 (Ridge) regularization
    # solver: Optimization algorithm (liblinear supports both L1 and L2, saga is newer)
    # class_weight: 'balanced' adjusts weights inversely proportional to class frequency
    param_dist_logreg = {
        "logreg__C": loguniform(1e-4, 1e3),  # Regularization strength
        "logreg__penalty": ["l1", "l2"],  # Regularization type
        "logreg__solver": ["liblinear", "saga"],  # Algorithm for optimization
        "logreg__class_weight": ["balanced", None],  # Handle class imbalance
    }

    # StratifiedKFold maintains class distribution in each fold (important for imbalanced data)
    cv_log = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Hyperparameter tuning for Logistic Regression with RandomizedSearchCV
    # Uses F1-score (harmonic mean of precision and recall) as objective
    # Better than accuracy for imbalanced classification problems
    print("\n--- Tuning Logistic Regression ---")
    rs_logreg = RandomizedSearchCV(
        estimator=pipeline_logreg,
        param_distributions=param_dist_logreg,
        n_iter=200,  # Number of random combinations to sample
        scoring="f1",  # Use F1-score instead of accuracy (better for imbalanced data)
        cv=cv_log,  # StratifiedKFold to preserve class distribution
        random_state=random_state,
        n_jobs=-1,  # Parallel processing
        verbose=0,
    )
    rs_logreg.fit(X_train_log, y_train_log)

    print(f"\n{'='*50}")
    print("LOGISTIC REGRESSION - Best parameters:")
    for k, v in rs_logreg.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV F1: {rs_logreg.best_score_:.4f}")
    print(f"{'='*50}")

    # Generate predictions and probability estimates on test set
    y_pred_log = rs_logreg.predict(X_test_log)  # Hard class predictions (0 or 1)
    y_prob_log = rs_logreg.predict_proba(X_test_log)[:, 1]  # Probability of positive class

    # Calculate classification metrics
    acc = accuracy_score(y_test_log, y_pred_log)  # Overall correctness
    f1 = f1_score(y_test_log, y_pred_log)  # Harmonic mean of precision and recall
    cm = confusion_matrix(y_test_log, y_pred_log)  # 2x2 matrix of prediction outcomes
    tn, fp, fn, tp = cm.ravel()  # Extract individual metrics from confusion matrix

    # Print comprehensive evaluation results
    print(f"\n{'='*50}")
    print("TEST SET METRICS")
    print(f"{'='*50}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"{'='*50}")
    print("\nCONFUSION MATRIX BREAKDOWN:")
    print(f"{'='*50}")
    print(f"  True  Positives (TP) : {tp}  (correctly predicted survived)")
    print(f"  True  Negatives (TN) : {tn}  (correctly predicted did not survive)")
    print(f"  False Positives (FP) : {fp}  (predicted survived, actually did not)")
    print(f"  False Negatives (FN) : {fn}  (predicted did not survive, actually survived)")
    print(f"{'='*50}")
    print("\nFull Classification Report:\n")
    # Generates precision, recall, F1-score for each class
    print(classification_report(y_test_log, y_pred_log, target_names=["Not Survived", "Survived"]))

    # Generate visualizations of predictions and confusion matrix
    plot_logreg_predictions(y_test_log, y_pred_log, y_prob_log, acc, f1, colors, output_dir)
    plot_confusion_matrix(cm, tn, fp, fn, tp, output_dir)

    # Return comprehensive results dictionary for reporting and analysis
    return {
        "accuracy": acc,  # Overall correctness rate
        "f1": f1,  # Balanced metric for imbalanced classification
        "tp": tp,  # True positives (correctly identified survivors)
        "tn": tn,  # True negatives (correctly identified non-survivors)
        "fp": fp,  # False positives (false alarms)
        "fn": fn,  # False negatives (missed survivals)
        "minority_count": minority_count,  # Size of minority class after balancing
    }