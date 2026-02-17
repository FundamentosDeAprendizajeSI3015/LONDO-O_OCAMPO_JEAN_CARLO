"""
Data Quality Assessment Module
Performs statistical checks to evaluate feature usefulness,
redundancy, and potential issues before modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_zero_variance(df):
    """
    Identify features with zero variance (constant columns).
    These features provide no information for modeling.
    """
    zero_var_cols = [col for col in df.columns if df[col].nunique() == 1]
    
    print("Zero Variance Features:")
    print(zero_var_cols)
    
    return zero_var_cols

def check_low_variance(df, threshold=0.01):
    """
    Identify near-constant features based on variance threshold.
    """
    variances = df.var(numeric_only=True)
    low_var_cols = variances[variances < threshold].index.tolist()
    
    print(f"Low Variance Features (variance < {threshold}):")
    print(low_var_cols)
    
    return low_var_cols

def correlation_analysis(df):
    """
    Compute and visualize correlation matrix for numeric features.
    """
    corr_matrix = df.corr(numeric_only=True)
    
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.show()
    
    return corr_matrix

def detect_high_correlation(corr_matrix, threshold=0.9):
    """
    Identify highly correlated feature pairs.
    """
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                high_corr_pairs.append((colname, corr_matrix.columns[j]))
    
    print(f"Highly Correlated Feature Pairs (>|{threshold}|):")
    print(high_corr_pairs)
    
    return high_corr_pairs