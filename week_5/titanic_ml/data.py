"""
Data loading and preprocessing module for the Titanic ML project.

This module provides functions to:
1. Load raw Titanic dataset from CSV
2. Clean missing values and handle data quality issues
3. Engineer new features from existing data
4. Encode categorical variables for machine learning models
5. Print comprehensive data summaries and statistics

Key preprocessing steps include:
- Missing value imputation (age by class/sex, embarked port)
- Feature engineering (family size, cabin deck, title extraction)
- Feature binning (age groups, fare quartiles)
- Categorical encoding using LabelEncoder

Functions:
    load_data: Load CSV data into DataFrame
    print_data_overview: Display raw data statistics and structure
    clean_and_engineer_features: Apply all preprocessing transformations
    print_clean_overview: Display cleaned data statistics
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(data_path):
    """
    Load Titanic dataset from CSV file.
    
    Args:
        data_path (str or Path): Full path to the CSV file
    
    Returns:
        pd.DataFrame: Raw dataset with all original columns and rows
    """
    return pd.read_csv(data_path)

def print_data_overview(df_raw: pd.DataFrame) -> None:
    """
    Display comprehensive overview of raw dataset.
    
    Prints the following information:
    - Dataset dimensions (rows, columns)
    - First 5 sample records
    - Data type of each column
    - Descriptive statistics (mean, std, quartiles)
    - Count of missing values per column
    - Percentage of missing values per column
    
    Args:
        df_raw (pd.DataFrame): Raw input dataset to analyze
    
    Returns:
        None: Only prints to console
    """
    print(f"\nDataset shape: {df_raw.shape}")
    print(f"\nFirst 5 rows:\n{df_raw.head()}")
    print(f"\nData types:\n{df_raw.dtypes}")
    print(f"\nBasic statistics:\n{df_raw.describe()}")
    print(f"\nMissing values:\n{df_raw.isnull().sum()}")
    print(f"\nMissing values (%):\n{(df_raw.isnull().sum() / len(df_raw) * 100).round(2)}")


def clean_and_engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data and engineer new features for improved model performance.
    
    This function applies the following transformations:
    1. Imputes missing age values using median by passenger class and sex
    2. Imputes remaining missing ages with global median
    3. Fills missing embarkation ports with mode (most common port)
    4. Extracts deck letter from cabin information
    5. Extracts title from passenger names
    6. Creates family size and alone status features
    7. Creates age and fare binning features
    8. Encodes all categorical variables as integers
    9. Removes original non-encoded columns
    
    Args:
        df_raw (pd.DataFrame): Raw dataset with original features
    
    Returns:
        pd.DataFrame: Cleaned and engineered dataset ready for modeling
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df_raw.copy()

    # Handle missing age values using group-wise median imputation
    # Strategy: First fill by Pclass and Sex groups, then use global median for remaining NaN
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    # Fill missing embarkation ports with the most frequently occurring port
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Extract cabin deck letter (A-F, T, U for unknown)
    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U")

    # Extract title from passenger names using regex pattern
    # Pattern looks for text after comma and space, followed by a dot
    df["Title"] = df["Name"].str.extract(r",\s*(\w+)\.", expand=False)
    # Map extracted titles to main categories; group rare titles as "Other"
    title_map = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
    df["Title"] = df["Title"].map(title_map).fillna("Other")

    # Engineer family-related features
    # FamilySize: Total family members (self + siblings/spouse + parents/children)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    # IsAlone: Binary indicator for traveling solo
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # Cut age into ordinal categories (Child < 12, Teen 12-18, Adult 18-35, etc.)
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"],
    )
    # Quartile-based fare binning (divide into 4 equal-frequency groups)
    df["FareBin"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "VeryHigh"])

    # Encode categorical variables to integers for ML models
    # LabelEncoder assigns 0, 1, 2... to unique categories in alphabetical order
    df["Sex_enc"] = LabelEncoder().fit_transform(df["Sex"])  # Female=0, Male=1
    df["Embarked_enc"] = LabelEncoder().fit_transform(df["Embarked"])  # C, Q, S ports
    df["Title_enc"] = LabelEncoder().fit_transform(df["Title"])  # Various titles
    df["Deck_enc"] = LabelEncoder().fit_transform(df["Deck"])  # Deck letters
    df["AgeBin_enc"] = LabelEncoder().fit_transform(df["AgeBin"])  # Age categories

    # Drop original columns that are now encoded or no longer needed for modeling
    cols_to_drop = [
        "PassengerId",  # Not a predictor for survival
        "Name",  # Already extracted title from name
        "Ticket",  # Unlikely to have predictive power
        "Cabin",  # Already extracted deck information
        "Embarked",  # Already encoded as Embarked_enc
        "Sex",  # Already encoded as Sex_enc
        "Title",  # Already encoded as Title_enc
        "Deck",  # Already encoded as Deck_enc
        "AgeBin",  # Already encoded as AgeBin_enc
        "FareBin",  # Binned for visualization, not needed for modeling
    ]
    return df.drop(columns=cols_to_drop)

def print_clean_overview(df_clean: pd.DataFrame) -> None:
    """
    Display summary of cleaned and engineered dataset.
    
    Prints the following information:
    - Final dataset dimensions (rows, columns)
    - Complete list of column names (all encoded and engineered features)
    - Count of missing values per column (should be zero after cleaning)
    - First 5 rows of cleaned data (ready for modeling)
    
    Args:
        df_clean (pd.DataFrame): Thoroughly cleaned dataset after preprocessing
    
    Returns:
        None: Only prints to console
    """
    print(f"\nCleaned dataset shape: {df_clean.shape}")
    print(f"\nCleaned columns: {df_clean.columns.tolist()}")
    print(f"\nMissing values after cleaning:\n{df_clean.isnull().sum()}")
    print(f"\nCleaned data sample:\n{df_clean.head()}")