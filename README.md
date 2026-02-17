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
# Project — Insight X: Machine Learning for Data Exfiltration Detection

**Project Overview**

Insight X is the final academic project developed for the course, focused on demonstrating the complete Machine Learning lifecycle through a cybersecurity use case.
The selected problem addresses the detection of data exfiltration attempts in corporate network environments using Machine Learning techniques.
The motivation behind the project is based on real-world scenarios where:

**Threats are not always external.**

- Data leaks may originate from users with legitimate access.

- Malicious behavior often attempts to resemble normal activity.

- Static rule-based systems struggle to adapt to evolving behaviors.

- The project emphasizes conceptual understanding, justification of technical decisions, and realistic system design rather than maximizing predictive performance.

## Problem Definition

**The problem is defined as:**

Detecting anomalous patterns in corporate network traffic that may indicate potential attempts of sensitive data exfiltration.

## Key characteristics of the problem:

- Primarily unsupervised

- Complete and reliable labels are rarely available

- Many exfiltration attempts remain undetected

- Attack strategies evolve over time

- Normal behavior changes dynamically

**Machine Learning is justified because:**

- Rule-based systems do not scale well in dynamic environments

- Legitimate traffic patterns are complex and variable

- ML allows modeling normal behavior distributions instead of predefined rules

**Success criteria include:**

- Low false positive rate

- Detection of meaningful anomalous behaviors

- Temporal stability

- Practical usefulness for human analysts

- Integration into a realistic monitoring workflow

## Data Source and Dataset

The project uses the NSL-KDD dataset, selected as an academic benchmark for network intrusion analysis.

## Reasons for selection:

- Represents network connection behavior

- Allows studying normal vs attack patterns

- Widely used for academic experimentation

- Suitable as a proxy dataset despite known limitations

## Dataset characteristics:

- 41 input features

- Each row represents a network connection

- Mixed numerical and categorical variables

- Dimensionality increases after encoding

- Although labels exist, they are not used during training, maintaining an unsupervised approach.

- Data quality considerations discussed in the project include:

- Dataset bias and imbalance

- Noise and redundancy

- Artificial attack patterns

- Lack of modern insider threat representation

- Concept drift limitations in static datasets

- A central idea of the project is:

- Effective dimensionality depends on data representation.

## Data Preparation and Representation

The preprocessing stage focuses on constructing a meaningful representation of behavior rather than aggressively cleaning the dataset.

**Main steps include:**

Basic cleaning and normalization

**Feature scaling**

**Encoding categorical variables**

**Feature engineering**

**Dimensionality reduction analysis**

A key conceptual decision was avoiding blind removal of outliers, since:

Data exfiltration is not a single event, but a behavioral pattern.

**Removing anomalous points without analysis may eliminate the signal that the model needs to detect.**

## Modeling Approach

**The selected model is Isolation Forest, chosen because:**

- It is an unsupervised algorithm

- Performs well in high-dimensional spaces

- Is computationally efficient

- Does not rely directly on distance metrics

- Has practical industry applications in anomaly detection

## Training assumptions:

The model is trained primarily on normal historical traffic.

The output is an anomaly score, where extreme values indicate unusual behavior.

## Evaluation Strategy

- Evaluation does not rely solely on accuracy metrics.

- Instead, the project considers:

- Distribution of anomaly scores

- False positive behavior

- Simulated attack scenarios

- Conceptual validation by analysts

- Temporal stability of detections

- This reflects real-world anomaly detection settings where ground truth is incomplete.

 ## Deployment Perspective

The project proposes a conceptual deployment pipeline:

- Network traffic capture

- Data preprocessing

- Model inference (anomaly scoring)

- Alert generation

- Human analyst validation

**An important design principle is:**

- Machine Learning supports analysts; it does not replace them.

- The system requires:

- Continuous monitoring

- Periodic retraining

- Adaptation to behavioral changes over time

- A central conclusion of the project states:

- A model that is not updated will eventually detect the future as an anomaly.
---

# **Technologies Used**

- Python

- NumPy

- Pandas

- Matplotlib

- Seaborn

- Scikit-learn

- Babyplots
