"""
Main Pipeline Module
Orchestrates the complete machine learning pipeline for network intrusion detection.
Handles data loading, preprocessing, model training, and evaluation.
"""

from data_loader import load_data
from preprocessing import prepare_training_data, prepare_test_data
from model import train_model, evaluate_model
from sklearn.metrics import classification_report

# File paths for training and test datasets
TRAIN_PATH = "../data/KDDTrain+.txt"
TEST_PATH = "../data/KDDTest+.txt"

def main():
    """
    Execute the complete machine learning pipeline.
    
    Workflow:
        1. Load raw datasets from files
        2. Preprocess training data (normalization, encoding)
        3. Train anomaly detection model
        4. Preprocess test data with training parameters
        5. Evaluate model performance with classification metrics
    """
    # Load raw training and test datasets
    print("Loading data...")
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    
    # Preprocess training data: filter, encode categorical features, scale
    print("Preparing training data...")
    X_train, scaler, train_columns = prepare_training_data(train_df)
    
    # Train Isolation Forest model on normal traffic
    print("Training model...")
    model = train_model(X_train)
    
    # Preprocess test data using training scaler and column alignment
    print("Preparing test data...")
    X_test = prepare_test_data(test_df, scaler, train_columns)
    
    # Generate predictions and anomaly scores
    print("Evaluating model...")
    predictions, scores = evaluate_model(model, X_test)
    
    # Convert labels: 1 for normal, -1 for attack (binary classification)
    true_labels = test_df['label'].apply(lambda x: 1 if x == 'normal' else -1)
    
    # Display classification metrics (precision, recall, F1-score)
    print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    main()