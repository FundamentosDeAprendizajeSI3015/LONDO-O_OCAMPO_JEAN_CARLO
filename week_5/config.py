"""
Configuration module for the Titanic ML project.

This module centralizes all configuration settings including file paths,
directory structures, random seeds, and visualization color schemes.
It ensures consistent configuration across the entire project.
"""

from pathlib import Path

# Get the absolute path of the project root directory
BASE_DIR = Path(__file__).resolve().parent

# List of candidate paths for the Titanic dataset
# Enables flexibility in data location for different environments
DATA_CANDIDATES = [
    BASE_DIR / "Titanic-Dataset.csv",
    BASE_DIR / "data" / "Titanic-Dataset.csv",
]

# Define the directory where model results and outputs will be stored
RESULTS_DIR = BASE_DIR / "results"

# Subdirectory for saving generated plots and visualizations
IMAGES_DIR = RESULTS_DIR / "images"

# Random seed for reproducibility across machine learning models
# Ensures consistent results across different runs
RANDOM_STATE = 42

# Color palette dictionary for consistent visualization styling
# Maps data categories to their corresponding hex color codes
COLORS = {
    "train": "#3498db",      # Blue for training data
    "test": "#e74c3c",       # Red for test data
    "ridge": "#2ecc71",      # Green for Ridge regression model
    "lasso": "#9b59b6",      # Purple for Lasso regression model
    "pos": "#27ae60",        # Dark green for positive predictions
    "neg": "#c0392b",        # Dark red for negative predictions
}