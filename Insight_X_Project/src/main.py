"""
Main Pipeline Module
Orchestrates the complete machine learning pipeline for network intrusion detection.
Handles data loading, preprocessing, model training, evaluation, and visualization.
"""

from data_loader import load_data
from preprocessing import prepare_training_data, prepare_test_data
from model import train_model, evaluate_model
from sklearn.metrics import classification_report

from eda import basic_dataset_overview
from data_quality import (
    check_zero_variance,
    check_low_variance,
    correlation_analysis,
    detect_high_correlation
)

from visualization import (
    plot_label_distribution,
    plot_feature_distribution,
    plot_anomaly_score_distribution,
    plot_score_by_class,
    plot_pca_projection
)

# File paths for training and test datasets
TRAIN_PATH = "../data/KDDTrain+.txt"
TEST_PATH = "../data/KDDTest+.txt"


def main():
    """
    Execute the complete machine learning pipeline.

    Workflow:
        1. Load datasets
        2. Perform EDA and data quality analysis
        3. Preprocess training data
        4. Train Isolation Forest model
        5. Preprocess test data
        6. Evaluate model
        7. Visualize results
    """

    # =========================
    # 1. Load Data
    # =========================
    print("Loading data...")
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # =========================
    # 2. EDA & Data Quality
    # =========================
    print("Performing EDA...")
    basic_dataset_overview(train_df, "Training Data")
    plot_label_distribution(train_df)

    print("Running data quality audit...")
    features_only = train_df.drop(columns=['label', 'difficulty'])

    zero_var = check_zero_variance(features_only)
    low_var = check_low_variance(features_only)

    corr_matrix = correlation_analysis(features_only)
    high_corr = detect_high_correlation(corr_matrix)

    # =========================
    # 3. Preprocessing
    # =========================
    print("Preparing training data...")
    X_train, scaler, train_columns = prepare_training_data(train_df)

    # =========================
    # 4. Model Training
    # =========================
    print("Training model...")
    model = train_model(X_train)

    # =========================
    # 5. Test Preprocessing
    # =========================
    print("Preparing test data...")
    X_test = prepare_test_data(test_df, scaler, train_columns)

    # =========================
    # 6. Model Evaluation
    # =========================
    print("Evaluating model...")
    predictions, scores = evaluate_model(model, X_test)

    # Convert labels: 1 for normal, -1 for attack
    true_labels = test_df['label'].apply(lambda x: 1 if x == 'normal' else -1)

    print(classification_report(true_labels, predictions))

    # =========================
    # 7. Visualizations
    # =========================
    print("Generating visualizations...")

    # Feature distributions
    plot_feature_distribution(train_df, 'src_bytes')
    plot_feature_distribution(train_df, 'dst_bytes')
    plot_feature_distribution(train_df, 'serror_rate')

    # Anomaly score analysis
    plot_anomaly_score_distribution(scores)
    plot_score_by_class(scores, true_labels.values)

    # PCA projection
    plot_pca_projection(X_test, true_labels.values)


if __name__ == "__main__":
    main()