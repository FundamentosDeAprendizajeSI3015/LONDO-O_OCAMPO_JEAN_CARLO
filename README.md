# Fundamentals of Machine Learning – SI3015

**Student:** Jean Carlo Londoño Ocampo

This repository contains the activities developed during the course Fundamentals of Machine Learning, organized by weeks and practical projects.

The main focus of the course was data exploration, analysis, and preprocessing, understanding that the quality of data representation (X, y) is a critical step before training Machine Learning models.

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

The repository reflects the actual workflow followed during the course, prioritizing the understanding of the data pipeline before model development.

---

## Week 2 — Wine Dataset Analysis

**Dataset:** Wine Dataset (scikit-learn)

### Objective

Introduce the basic Machine Learning workflow, from data collection to initial model evaluation in a supervised learning setting.

This dataset was selected because:
- It represents a multiclass supervised classification problem
- It contains more features than Iris, allowing exploration of feature scaling, correlations, feature importance, and overfitting risks
- It remains small and manageable for academic purposes

### Activities Performed

- Dataset loading using scikit-learn
- Conversion into a pandas DataFrame
- Initial data exploration
- Class distribution analysis
- Train/Test split
- Feature scaling using StandardScaler
- Model training: Logistic Regression
- Model evaluation: Accuracy, Classification Report
- Cross-validation to reduce overfitting risk
- Feature importance analysis using model coefficients
- Visualization of most relevant features

### Concepts Covered

- X, y representation
- Feature scaling
- Model generalization
- Overfitting
- Basic model interpretability

---

## Week 3 — Iris Dataset: Data Preprocessing Pipeline

**Dataset:** Iris Dataset (scikit-learn)

### Objective

Build a complete data exploration and preprocessing pipeline before training Machine Learning models. The main objective was understanding that model performance strongly depends on the quality of the prepared dataset.

### Activities Performed

- Dataset loading and DataFrame construction
- Initial exploration (EDA): general dataset information, descriptive statistics, missing value checks, feature uniqueness analysis
- Exploratory visualization: pairplots, correlation heatmap, 3D visualization using Babyplots
- Outlier detection using Z-score
- Feature standardization (StandardScaler)
- Pattern discovery using PCA
- Stratified Train/Test split
- Export of processed datasets in parquet format

### Concepts Covered

- Data quality assessment
- Geometric interpretation of datasets
- Dimensionality reduction
- Separation between preprocessing and modeling
- Reproducible data pipelines

---

## Week 4 — Movies Dataset: Data Exploration and Transformation

**Dataset:** movies.csv

### Objective

Develop a complete data cleaning and transformation workflow using a dataset with real-world data quality issues. This laboratory simulated a more realistic data preprocessing scenario.

### Activities Performed

- **Data Cleaning and Normalization:** column name normalization, data type conversion, removal of currency symbols, handling missing values
- **Descriptive Statistics:** central tendency measures (mean, median, mode), dispersion measures (standard deviation, variance, range, IQR), position measures (quartiles, min/max values)
- **Outlier Detection:** IQR-based outlier detection, removal of extreme values
- **Visualization:** histograms, scatter plots
- **Feature Transformation:** Label Encoding, One-Hot Encoding, binary encoding
- **Correlation Analysis:** removal of highly correlated features
- **Scaling and Transformations:** MinMaxScaler, StandardScaler, logarithmic transformations for skewed variables
- **Output Generation:** cleaned dataset, transformed dataset ready for Machine Learning models

### Concepts Covered

- Data quality challenges in real datasets
- Basic feature engineering
- Statistical noise reduction
- Preparation of tabular data for supervised learning

---

## Week 5 — Titanic Dataset: Regression and Classification Pipelines

**Dataset:** Titanic Dataset (CSV)

### Objective

Apply a complete, modular Machine Learning pipeline covering data cleaning, feature engineering, and model training for both regression and classification tasks on the same dataset.

### Activities Performed

- **Data Loading and Exploration:** dataset shape analysis, data types inspection, descriptive statistics, missing value profiling (counts and percentages)
- **Missing Value Imputation:** age imputation using group-wise median by passenger class and sex, embarkation port imputation using mode
- **Feature Engineering:** extraction of cabin deck letter, title extraction from passenger names using regex, family size computation (SibSp + Parch + 1), binary "IsAlone" indicator, age binning into ordinal categories (Child, Teen, Adult, MiddleAge, Senior), fare quartile binning
- **Categorical Encoding:** LabelEncoder applied to Sex, Embarked, Title, Deck, and AgeBin columns
- **Linear Regression (Fare Prediction):** outlier removal at the 99th percentile, Ridge and Lasso pipelines with StandardScaler, hyperparameter tuning via RandomizedSearchCV with 5-fold cross-validation (100 iterations, log-uniform alpha distribution), evaluation using R² and MAE on test set
- **Logistic Regression (Survival Prediction):** class imbalance analysis, random undersampling of the majority class, logistic regression pipeline with StandardScaler, hyperparameter tuning via RandomizedSearchCV with StratifiedKFold (200 iterations, optimizing F1-score), evaluation using Accuracy, F1-score, and confusion matrix breakdown (TP, TN, FP, FN)
- **Visualization:** EDA plots, correlation heatmap, train/test distribution comparison, prediction scatter plots, actual vs predicted plots, class balance comparison, confusion matrix heatmap
- **Modular Code Structure:** separate modules for data processing (`data.py`), model training (`models.py`), plotting (`plots.py`), file I/O utilities (`io_utils.py`), and orchestration (`pipeline.py`)

### Concepts Covered

- Modular ML project organization
- Group-wise imputation strategies
- Feature engineering from raw text and categorical data
- Regularized regression (Ridge L2 / Lasso L1) with hyperparameter search
- Class imbalance handling via undersampling
- Stratified cross-validation for classification
- Comparison of regression vs classification workflows on the same dataset

---

## Week 6 — Decision Trees and Ensembles: Random Forest vs Gradient Boosting

**Dataset:** NSL-KDD (merged train + test)

### Objective

Compare tree-based ensemble methods (Random Forest and Gradient Boosting) on a network intrusion detection task, with an emphasis on evaluating model performance under extreme data reduction.

### Activities Performed

- **Data Loading:** loaded KDDTrain+ and KDDTest+ files, merged and shuffled into a single DataFrame (148K+ records, 43 columns)
- **Basic EDA:** class distribution analysis (normal vs attack types), visualization of label frequencies
- **Data Transformation:** conversion to binary classification (normal = 0, attack = 1), removal of label and difficulty columns, One-Hot Encoding for categorical features (protocol_type, service, flag)
- **Extreme Reduction Experiment:** used only 5% of the full dataset for training, split the remaining 95% equally into validation and test sets, stratified splitting to preserve class distribution
- **Feature Scaling:** StandardScaler applied to train, validation, and test sets (fit on train only)
- **Random Forest:** trained with 200 estimators, evaluated on both validation and test sets
- **Gradient Boosting:** trained with 100 estimators and learning rate of 0.1, evaluated on both validation and test sets
- **Evaluation:** accuracy, full classification report (precision, recall, F1-score), confusion matrix heatmaps for each model on each split
- **Feature Importance:** top 15 feature importances from Random Forest visualized as a horizontal bar chart
- **Tree Visualization:** first tree of the Random Forest plotted with a depth limit of 3 for interpretability

### Concepts Covered

- Binary classification from multiclass labels
- One-Hot Encoding for high-cardinality categorical variables
- Train/Validation/Test three-way split strategy
- Behavior of ensembles under severe data scarcity (5% training)
- Random Forest vs Gradient Boosting trade-offs
- Feature importance extraction and interpretation
- Decision tree visualization for model explainability

---

## Week 7 — Overfitting, Underfitting, and Mitigation Techniques

**Material:** Course lecture — *Lectura 7: Problemas comunes (overfitting/underfitting) y técnicas de mitigación*

### Objective

Study the theoretical foundations of overfitting and underfitting in Machine Learning, understand their root causes, and learn the main mitigation techniques.

### Topics Covered

- **Overfitting vs Underfitting vs Optimal Fit:** definitions, behavior on training vs test data, and the bias-variance trade-off
- **Causes of Overfitting:** excessively complex models (deep trees, large neural networks), insufficient training data (spurious correlations, curse of dimensionality, missing edge cases), excessive training iterations (overtraining beyond the generalization inflection point)
- **Mitigation Techniques:**
  - **Regularization (L1 and L2):** Ridge (L2) shrinks weights to small values; Lasso (L1) drives some weights to zero enabling automatic feature selection
  - **Early Stopping:** monitoring validation loss and halting training at the inflection point where validation error starts to rise
  - **Data Augmentation:** geometric and photometric transforms for images, synonym replacement and back-translation for text, noise injection and pitch shifting for audio; acts as implicit regularization by smoothing the decision boundary
  - **Transfer Learning:** instance re-weighting, feature transfer from pre-trained models, relational rule transfer, domain adaptation via shared projections (PCA)
  - **Cross-Validation (K-Fold):** eliminates selection bias from single splits, detects model instability through fold variance, enables robust hyperparameter tuning by selecting configurations with the best average performance

### Concepts Covered

- Bias-variance trade-off
- Loss landscape navigation (broad vs narrow minima)
- Empirical vs theoretical data distribution mismatch
- Regularization as constrained optimization
- Double descent phenomenon in over-parameterized models
- Cross-validation as both evaluation and model selection tool

---

## Week 8 — FIRE-UdeA: Financial Risk Prediction Model

**Dataset:** Synthetic financial dataset — `dataset_sintetico_FIRE_UdeA_realista.csv`

### Objective

Build a classification pipeline to predict financial risk in academic units of Universidad de Antioquia, applying tree-based models with hyperparameter optimization and comprehensive evaluation.

### Activities Performed

- **Data Loading and EDA:** loaded synthetic financial dataset with 15 features (income, expenses, liquidity, cash days, CFO, revenue source participation, HHI diversification index, debt ratio, income trends, GP ratio), class distribution analysis for the binary risk label, dataset info and descriptive statistics
- **Visualization:** overlaid histograms per feature colored by risk class, correlation heatmap including label correlations
- **Data Cleaning:** median imputation for missing values across all predictive features, exclusion of non-predictive columns (year, unit name)
- **Train/Test Split:** 75/25 stratified split to maintain class proportions
- **Decision Tree:** GridSearchCV with StratifiedKFold (5 folds), hyperparameter search over max_depth, min_samples_split, min_samples_leaf, and criterion (gini/entropy), optimized for F1-score
- **Random Forest:** GridSearchCV with StratifiedKFold (5 folds), hyperparameter search over n_estimators, max_depth, min_samples_split, min_samples_leaf, and max_features, optimized for F1-score
- **Gradient Boosting:** trained with fixed hyperparameters (100 estimators, max_depth=3, learning_rate=0.1)
- **Model Evaluation:** train and test accuracy, precision, recall, F1-score, ROC-AUC for each model, confusion matrix heatmaps, comparative ROC curves for all three models
- **Feature Importance:** horizontal bar chart showing Random Forest feature importances to identify the most predictive financial indicators

### Concepts Covered

- Financial risk classification as a supervised learning problem
- Synthetic data generation for academic research
- GridSearchCV with stratified cross-validation
- Comparison of Decision Tree, Random Forest, and Gradient Boosting classifiers
- ROC-AUC analysis for model comparison
- Feature importance for domain interpretation in financial contexts
- End-to-end ML pipeline: EDA → cleaning → training → evaluation → visualization

---

## Project — Insight X: Machine Learning for Data Exfiltration Detection

### Project Overview

Insight X is the final academic project developed for the course, focused on demonstrating the complete Machine Learning lifecycle through a cybersecurity use case. The selected problem addresses the detection of data exfiltration attempts in corporate network environments using Machine Learning techniques. The motivation behind the project is based on real-world scenarios where threats are not always external, data leaks may originate from users with legitimate access, malicious behavior often attempts to resemble normal activity, and static rule-based systems struggle to adapt to evolving behaviors.

The project emphasizes conceptual understanding, justification of technical decisions, and realistic system design rather than maximizing predictive performance.

### Problem Definition

Detecting anomalous patterns in corporate network traffic that may indicate potential attempts of sensitive data exfiltration.

Key characteristics of the problem: primarily unsupervised, complete and reliable labels are rarely available, many exfiltration attempts remain undetected, attack strategies evolve over time, and normal behavior changes dynamically.

Machine Learning is justified because rule-based systems do not scale well in dynamic environments, legitimate traffic patterns are complex and variable, and ML allows modeling normal behavior distributions instead of predefined rules.

Success criteria include low false positive rate, detection of meaningful anomalous behaviors, temporal stability, practical usefulness for human analysts, and integration into a realistic monitoring workflow.

### Data Source and Dataset

The project uses the **NSL-KDD dataset**, selected as an academic benchmark for network intrusion analysis. It represents network connection behavior, allows studying normal vs attack patterns, is widely used for academic experimentation, and is suitable as a proxy dataset despite known limitations.

Dataset characteristics: 41 input features, each row represents a network connection, mixed numerical and categorical variables, and dimensionality increases after encoding. Although labels exist, they are not used during training, maintaining an unsupervised approach.

Data quality considerations discussed in the project include dataset bias and imbalance, noise and redundancy, artificial attack patterns, lack of modern insider threat representation, and concept drift limitations in static datasets.

### Data Preparation and Representation

The preprocessing stage focuses on constructing a meaningful representation of behavior rather than aggressively cleaning the dataset. Main steps include basic cleaning and normalization, feature scaling, encoding categorical variables, feature engineering, and dimensionality reduction analysis.

A key conceptual decision was avoiding blind removal of outliers, since data exfiltration is not a single event but a behavioral pattern, and removing anomalous points without analysis may eliminate the signal that the model needs to detect.

### Modeling Approach

The selected model is **Isolation Forest**, chosen because it is an unsupervised algorithm, performs well in high-dimensional spaces, is computationally efficient, does not rely directly on distance metrics, and has practical industry applications in anomaly detection. The model is trained primarily on normal historical traffic, and the output is an anomaly score where extreme values indicate unusual behavior.

### Evaluation Strategy

Evaluation does not rely solely on accuracy metrics. Instead, the project considers distribution of anomaly scores, false positive behavior, simulated attack scenarios, conceptual validation by analysts, and temporal stability of detections. This reflects real-world anomaly detection settings where ground truth is incomplete.

### Deployment Perspective

The project proposes a conceptual deployment pipeline: network traffic capture → data preprocessing → model inference (anomaly scoring) → alert generation → human analyst validation.

An important design principle is that Machine Learning supports analysts; it does not replace them. The system requires continuous monitoring, periodic retraining, and adaptation to behavioral changes over time.

---

## General Course Conclusions

Throughout the activities, the following insights were observed:

- Model performance heavily depends on data preprocessing quality.
- Visual exploration helps understand the geometric structure of datasets.
- Scaling and feature transformation are essential steps in ML workflows.
- Separating preprocessing from modeling improves reproducibility.
- Real-world datasets require significantly more cleaning than academic datasets.
- Modular code organization enables cleaner experimentation and iteration.
- Tree-based ensembles provide strong baselines with built-in feature importance.
- Overfitting mitigation requires a combination of techniques: regularization, cross-validation, and careful data management.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Babyplots
