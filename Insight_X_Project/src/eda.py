"""
Exploratory Data Analysis (EDA) Module
Performs statistical and structural analysis of the NSL-KDD dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def basic_dataset_overview(df, name="Dataset"):
    """
    Print basic information about dataset.
    """
    print(f"\n{name} Shape: {df.shape}")
    print(f"\n{name} Info:")
    print(df.info())
    print(f"\n{name} Label Distribution:")
    print(df['label'].value_counts())

def plot_label_distribution(df):
    """
    Plot distribution of normal vs attack labels.
    """
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df)
    plt.title("Label Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_numeric_distribution(df, column):
    """
    Plot histogram of a numeric feature.
    """
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], bins=50, kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()