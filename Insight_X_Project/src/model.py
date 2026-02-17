"""
Anomaly Detection Model Module
Trains and evaluates Isolation Forest model for network intrusion detection.
"""

from sklearn.ensemble import IsolationForest

def train_model(X_train):
    """
    Train Isolation Forest model on normal traffic data.
    
    Args:
        X_train (np.ndarray): Preprocessed training data
        
    Returns:
        IsolationForest: Trained anomaly detection model
    """
    # Initialize Isolation Forest with hyperparameters
    # n_estimators: number of base estimators
    # contamination: expected fraction of outliers (anomalies)
    # random_state: for reproducibility
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )
    
    # Fit model on training data
    model.fit(X_train)
    
    return model

def evaluate_model(model, X_test):
    """
    Generate predictions and anomaly scores for test data.
    
    Args:
        model (IsolationForest): Trained anomaly detection model
        X_test (np.ndarray): Preprocessed test data
        
    Returns:
        tuple: (predictions array with -1/1, anomaly scores)
    """
    # Predict: -1 indicates anomaly, 1 indicates normal
    predictions = model.predict(X_test)
    
    # Get anomaly scores (negative = more anomalous)
    scores = model.decision_function(X_test)
    
    return predictions, scores