"""
Data Preprocessing Module
Handles data transformation, normalization, and encoding for network intrusion detection.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Categorical features requiring one-hot encoding
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

def prepare_training_data(train_df):
    """
    Prepare training data for anomaly detection.
    Only normal traffic is used for fitting the model.
    """
    # Use only normal traffic to model baseline behavior
    normal_df = train_df[train_df['label'] == 'normal']

    X = normal_df.drop(columns=['label', 'difficulty'])

    X = pd.get_dummies(X, columns=CATEGORICAL_COLS)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, X.columns

def prepare_test_data(test_df, scaler, train_columns):
    """
    Prepare test data using training scaler and column alignment.
    
    Args:
        test_df (pd.DataFrame): Raw test dataset
        scaler: Fitted StandardScaler from training phase
        train_columns: Feature columns from training data
        
    Returns:
        np.ndarray: Scaled test data aligned with training features
    """
    # Remove non-feature columns
    X = test_df.drop(columns=['label', 'difficulty'])
    
    # Apply one-hot encoding with same categorical features
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS)
    
    # Ensure test data has exact same features as training
    # (fill missing with 0, remove extra columns)
    X = X.reindex(columns=train_columns, fill_value=0)
    
    # Apply trained scaler transformation to test data
    X_scaled = scaler.transform(X)
    
    return X_scaled