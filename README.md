# **Fundamentals of Machine Learning – SI3015**

**Student: Jean Carlo Londoño Ocampo**

This repository contains the activities developed during the course Fundamentals of Machine Learning, organized by weeks and practical projects.

The main focus of the course was data exploration, analysis, and preprocessing, understanding that the quality of data representation (X, y) is a critical step before training Machine Learning models.

---

# **Repository Objective**

The goal of this repository is to document the complete data preparation process for Machine Learning, including:

- Initial dataset exploration (EDA)

- Data quality evaluation

- Descriptive statistics

- Outlier detection

- Pattern visualization

- Variable transformation and encoding

- Feature scaling

- Construction of datasets ready for modeling

The repository reflects the actual workflow followed during the course, prioritizing the understanding of the data pipeline before model development.

---

# **Week 2 — Wine Dataset Analysis**
** Dataset**

Wine Dataset (scikit-learn)

## **Objective**

Introduce the basic Machine Learning workflow, from data collection to initial model evaluation in a supervised learning setting.

This dataset was selected because:

- It represents a multiclass supervised classification problem

- It contains more features than Iris, allowing exploration of:

- feature scaling

- correlations

- feature importance

- overfitting risks

- It remains small and manageable for academic purposes.

## **Activities Performed**

- Dataset loading using scikit-learn

- Conversion into a pandas DataFrame

- Initial data exploration

- Class distribution analysis

- Train/Test split

- Feature scaling using StandardScaler

## **Model training:**

- Logistic Regression

## **Model evaluation:**

- Accuracy

- Classification Report

- Cross-validation to reduce overfitting risk

- Feature importance analysis using model coefficients

- Visualization of most relevant features

## **Concepts Covered**

- X, y representation

- Feature scaling

- Model generalization

- Overfitting

- Basic model interpretability

---

# **Week 3 — Iris Dataset: Data Preprocessing Pipeline**
** Dataset**

Iris Dataset (scikit-learn)

## **Objective**

Build a complete data exploration and preprocessing pipeline before training Machine Learning models.

The main objective was understanding that model performance strongly depends on the quality of the prepared dataset.

## **Activities Performed**

- Dataset loading and DataFrame construction

- Initial exploration (EDA):

- General dataset information

- Descriptive statistics

- Missing value checks

- Feature uniqueness analysis

- Exploratory visualization:

- Pairplots

- Correlation heatmap

- 3D visualization using Babyplots

- Outlier detection using Z-score

- Feature standardization (StandardScaler)

- Pattern discovery using PCA

- Stratified Train/Test split

- Export of processed datasets in parquet format

## **Concepts Covered**

- Data quality assessment

- Geometric interpretation of datasets

- Dimensionality reduction

- Separation between preprocessing and modeling

- Reproducible data pipelines

---

# **Week 4 — Movies Dataset: Data Exploration and Transformation**
**Dataset**

movies.csv

## **Objective**

Develop a complete data cleaning and transformation workflow using a dataset with real-world data quality issues.

This laboratory simulated a more realistic data preprocessing scenario.

## **Activities Performed**
- Data Cleaning and Normalization

- Column name normalization

- Data type conversion

- Removal of currency symbols

- Handling missing values

## **Descriptive Statistics**

- Central tendency measures:

- mean

- median

- mode

- Dispersion measures:

- standard deviation

- variance

- range

- IQR

- Position measures:

- quartiles

- minimum and maximum values

## **Outlier Detection**

- IQR-based outlier detection

- Removal of extreme values

## **Visualization**

- Histograms

- Scatter plots

## **Feature Transformation**

- Label Encoding

- One-Hot Encoding

- Binary encoding

## **Correlation Analysis**

- Removal of highly correlated features

## **Scaling and Transformations**

- MinMaxScaler

- StandardScaler

- Logarithmic transformations for skewed variables

## **Output Generation**

- Cleaned dataset

- Transformed dataset ready for Machine Learning models

## **Concepts Covered**

- Data quality challenges in real datasets

- Basic feature engineering

- Statistical noise reduction

- Preparation of tabular data for supervised learning

** General Course Conclusions**

- Throughout the activities, the following insights were observed:

- Model performance heavily depends on data preprocessing quality.

- Visual exploration helps understand the geometric structure of datasets.

- Scaling and feature transformation are essential steps in ML workflows.

- Separating preprocessing from modeling improves reproducibility.

- Real-world datasets require significantly more cleaning than academic datasets.

---

# **Technologies Used**

- Python

- NumPy

- Pandas

- Matplotlib

- Seaborn

- Scikit-learn

- Babyplots
