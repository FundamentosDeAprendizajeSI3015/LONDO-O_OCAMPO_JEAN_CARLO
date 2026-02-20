import warnings
from pathlib import Path

import numpy as np

from config import COLORS, DATA_CANDIDATES, IMAGES_DIR, RANDOM_STATE
from titanic_ml.data import clean_and_engineer_features, load_data, print_clean_overview, print_data_overview
from titanic_ml.models import run_linear_regression, run_logistic_regression
from titanic_ml.plots import plot_correlation, plot_eda, set_plot_theme

def resolve_data_path() -> Path:
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    options = "\n".join(str(p) for p in DATA_CANDIDATES)
    raise FileNotFoundError(f"No dataset found. Checked:\n{options}")

def run_pipeline() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_STATE)
    set_plot_theme()

    data_path = resolve_data_path()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SECTION 1: DATA LOADING & EXPLORATION")
    print("=" * 72)
    print(f"Using dataset: {data_path}")

    df_raw = load_data(data_path)
    print_data_overview(df_raw)
    plot_eda(df_raw, COLORS, IMAGES_DIR)

    print("\n" + "=" * 72)
    print("SECTION 2: DATA CLEANING & TRANSFORMATION")
    print("=" * 72)
    df_clean = clean_and_engineer_features(df_raw)
    print_clean_overview(df_clean)
    plot_correlation(df_clean, IMAGES_DIR)

    print("\n" + "=" * 72)
    print("PART 1: LINEAR REGRESSION - Predicting 'Fare'")
    print("=" * 72)
    lr_results = run_linear_regression(df_clean, COLORS, IMAGES_DIR, random_state=RANDOM_STATE)

    print("\n" + "=" * 72)
    print("PART 2: LOGISTIC REGRESSION - Predicting 'Survived'")
    print("=" * 72)
    log_results = run_logistic_regression(df_clean, COLORS, IMAGES_DIR, random_state=RANDOM_STATE)

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(
        f"""
LINEAR REGRESSION (Target: Fare)
  Ridge:  R2 = {lr_results['r2_ridge']:.4f}, MAE = {lr_results['mae_ridge']:.4f}, alpha = {lr_results['ridge_alpha']:.6f}
  Lasso:  R2 = {lr_results['r2_lasso']:.4f}, MAE = {lr_results['mae_lasso']:.4f}, alpha = {lr_results['lasso_alpha']:.6f}

LOGISTIC REGRESSION (Target: Survived)
  Accuracy  = {log_results['accuracy']:.4f}
  F1-Score  = {log_results['f1']:.4f}
  TP = {log_results['tp']}, TN = {log_results['tn']}, FP = {log_results['fp']}, FN = {log_results['fn']}
  Dataset balanced via undersampling ({log_results['minority_count']} samples per class)

Generated plots in: {IMAGES_DIR}
  01_eda.png
  02_correlation.png
  03_lr_train_test.png
  04_lr_predictions.png
  05_lr_actual_vs_pred.png
  06_class_balance.png
  07_logreg_predictions.png
  08_confusion_matrix.png
"""
    )
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)