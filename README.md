# Fundamentals of Machine Learning – SI3015  
**Student:** Jean Carlo Londoño Ocampo  

---

## Repository Overview  
This repository contains the activities developed during the course *Fundamentals of Machine Learning*, organized by weeks and practical projects.  

The main focus of the course was data exploration, analysis, and preprocessing, understanding that the quality of data representation (X, y) is a critical step before training Machine Learning models.  

---

## Repository Objective  
The goal of this repository is to document the complete data preparation process for Machine Learning, including:

- Initial dataset exploration (EDA)  
- Data quality evaluation  
- Descriptive statistics  
- Outlier detection  
- Pattern visualization  
- Variable transformation and encoding  
- Feature scaling  
- Construction of datasets ready for modeling  

The repository reflects a real-world workflow, prioritizing understanding the data pipeline before model development.

---

# Weekly Content

---

## Week 2 — Wine Dataset Analysis  
**Dataset:** Wine Dataset (scikit-learn)  

### Objective  
Introduce the basic Machine Learning workflow in a supervised classification setting.

### Activities  
- Dataset loading using scikit-learn  
- Conversion into a pandas DataFrame  
- Initial data exploration  
- Class distribution analysis  
- Train/Test split  
- Feature scaling using StandardScaler  
- Model training: Logistic Regression  
- Model evaluation: Accuracy, Classification Report  
- Cross-validation  
- Feature importance analysis  
- Visualization of relevant features  

### Concepts Covered  
- X, y representation  
- Feature scaling  
- Model generalization  
- Overfitting  
- Model interpretability  

---

## Week 3 — Iris Dataset: Data Preprocessing Pipeline  
**Dataset:** Iris Dataset (scikit-learn)  

### Objective  
Build a complete data exploration and preprocessing pipeline before training Machine Learning models.

### Activities  
- Dataset loading and DataFrame construction  
- Exploratory Data Analysis (EDA)  
- Descriptive statistics and missing value checks  
- Feature uniqueness analysis  
- Pairplots and correlation heatmap  
- 3D visualization using Babyplots  
- Outlier detection using Z-score  
- Feature standardization (StandardScaler)  
- PCA for pattern discovery  
- Stratified Train/Test split  
- Export of processed datasets in parquet format  

### Concepts Covered  
- Data quality assessment  
- Geometric interpretation of datasets  
- Dimensionality reduction  
- Separation between preprocessing and modeling  
- Reproducible pipelines  

---

## Week 4 — Movies Dataset: Data Exploration and Transformation  
**Dataset:** movies.csv  

### Objective  
Develop a complete data cleaning and transformation workflow using a dataset with real-world data quality issues.

### Activities  
- Column normalization and data type conversion  
- Removal of currency symbols  
- Handling missing values  
- Descriptive statistics (mean, median, mode, variance, IQR)  
- Outlier detection using IQR  
- Visualization (histograms, scatter plots)  
- Encoding (Label Encoding, One-Hot Encoding, binary encoding)  
- Correlation analysis  
- Feature scaling (MinMaxScaler, StandardScaler)  
- Logarithmic transformations  
- Generation of cleaned and transformed datasets  

### Concepts Covered  
- Data quality challenges  
- Feature engineering  
- Statistical noise reduction  
- Preparation of tabular data  

---

## Week 5 — Titanic Dataset: Regression and Classification Pipelines  
**Dataset:** Titanic Dataset (CSV)  

### Objective  
Apply a complete, modular Machine Learning pipeline for both regression and classification tasks.

### Activities  

#### Data Preparation  
- Missing value imputation (group-wise median)  
- Feature engineering (Deck, Title, Family Size, IsAlone, Age bins, Fare bins)  
- Label encoding for categorical variables  

#### Regression Task (Fare Prediction)  
- Ridge and Lasso regression  
- RandomizedSearchCV for hyperparameter tuning  
- Evaluation using R² and MAE  

#### Classification Task (Survival Prediction)  
- Class imbalance handling (undersampling)  
- Logistic Regression  
- Stratified cross-validation  
- Evaluation using Accuracy, F1-score, confusion matrix  

### Concepts Covered  
- Modular ML project organization  
- Regularization (L1 and L2)  
- Feature engineering from categorical data  
- Class imbalance handling  
- Regression vs classification workflows  

---

## Week 6 — Decision Trees and Ensembles  
**Dataset:** NSL-KDD  

### Objective  
Compare ensemble models under extreme data reduction conditions.

### Activities  
- Data loading and merging  
- Binary classification transformation  
- One-Hot Encoding  
- Train/Validation/Test split  
- Random Forest training (200 estimators)  
- Gradient Boosting training  
- Model evaluation (accuracy, precision, recall, F1-score)  
- Confusion matrix visualization  
- Feature importance analysis  
- Decision tree visualization  

### Concepts Covered  
- Ensemble methods  
- Feature importance  
- Model interpretability  
- Effects of limited training data  

---

## Week 7 — Overfitting, Underfitting, and Mitigation Techniques  

### Objective  
Understand generalization issues in Machine Learning and their solutions.

### Topics Covered  
- Overfitting vs underfitting vs optimal fit  
- Bias-variance trade-off  
- Causes of overfitting  
- Regularization (Ridge and Lasso)  
- Early stopping  
- Data augmentation  
- Transfer learning  
- Cross-validation (K-Fold)  

---

## Week 8 — FIRE-UdeA: Financial Risk Prediction Model  

### Objective  
Build a classification pipeline for financial risk prediction.

### Activities  
- Data loading and EDA  
- Missing value imputation  
- Feature selection  
- Train/Test split  
- Decision Tree with GridSearchCV  
- Random Forest with GridSearchCV  
- Gradient Boosting model  
- Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)  
- Confusion matrices and ROC curves  
- Feature importance analysis  

### Concepts Covered  
- Financial risk classification  
- Hyperparameter tuning  
- Model comparison  
- ROC-AUC analysis  
- End-to-end ML pipeline  

---

## Week 9 — FIRE-UdeA: Unsupervised Clustering Analysis  

### Objective  
Evaluate whether natural clusters align with known labels.

### Activities  
- Preprocessing and scaling  
- K-Means clustering  
- Elbow method  
- PCA visualization  
- DBSCAN clustering  
- Label alignment using Hungarian algorithm  
- Confusion matrix and classification report  

### Concepts Covered  
- Unsupervised learning evaluation  
- Clustering validation  
- PCA visualization  
- Density-based clustering  

---

## Week 10 — Clustering-Based Label Validation Pipeline  

### Objective  
Assess label validity using clustering and statistical analysis.

### Activities  
- K-Means, DBSCAN, HDBSCAN, Hierarchical clustering  
- PCA dimensionality reduction  
- External validation metrics (ARI, NMI, V-measure)  
- Conditional probability analysis P(class|cluster)  
- Bootstrap stability testing  
- Visualization (heatmaps, dendrograms, boxplots)  

### Key Finding  
Clustering results show ARI ≈ 0, indicating that labels do not reflect natural data structure.

### Concepts Covered  
- Clustering validation metrics  
- Stability analysis  
- Label quality assessment  
- Multi-algorithm comparison  

---

# Final Project — Insight X  

## Overview  
Insight X is a Machine Learning project focused on detecting data exfiltration attempts in network environments.

## Problem Definition  
Detect anomalous patterns in network traffic where labeled data is incomplete or unreliable.

## Approach  
- Unsupervised learning using Isolation Forest  
- Modeling normal behavior instead of predefined attack rules  

## Key Components  
- Data preprocessing and feature engineering  
- Dimensionality reduction analysis  
- Anomaly scoring  
- Clustering-based validation  
- Label correction experiments  
- Supervised vs unsupervised comparison  

## Results  
- High performance on original labels  
- Performance degradation after automatic relabeling  
- Reinforcement of unsupervised approach validity  

## Deployment Perspective  
Pipeline:  
network traffic → preprocessing → model inference → alert generation → analyst validation  

## Key Insight  
Machine Learning supports analysts but does not replace them.

---

# General Conclusions  

- Model performance depends heavily on preprocessing quality  
- Data visualization helps understand structure  
- Feature scaling is essential  
- Modular design improves reproducibility  
- Real datasets require extensive cleaning  
- Ensemble models provide strong baselines  
- Overfitting requires multiple mitigation strategies  
- Clustering helps validate data structure  
- Automatic label correction must be validated  
- High accuracy does not guarantee real-world performance  

---

# Technologies Used  

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- scikit-fuzzy  
- Babyplots  
