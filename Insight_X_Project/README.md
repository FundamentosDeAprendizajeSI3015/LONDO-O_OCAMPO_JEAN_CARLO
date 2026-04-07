# Network Anomaly Detection using Isolation Forest  
NSL-KDD Dataset – Machine Learning Lifecycle Project (Insight X)

**Author:** Jean Carlo Londoño Ocampo  
**Course:** Fundamentos de Aprendizaje Automático  
**Year:** 2026  

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Definition](#problem-definition)
- [Dataset](#dataset)
- [Data Exploration (EDA)](#data-exploration-eda)
- [Data Quality Audit](#data-quality-audit)
- [Feature Reduction](#feature-reduction)
- [Preprocessing](#preprocessing)
- [Model Selection](#model-selection)
- [Evaluation](#evaluation)
- [Extended Analysis](#extended-analysis)
- [Project Structure](#project-structure)
- [ML Lifecycle Covered](#ml-lifecycle-covered)
- [Future Improvements](#future-improvements)
- [Key Takeaways](#key-takeaways)
- [Generated Visualizations](#generated-visualizations)
- [Final Reflection](#final-reflection)

---

## Project Overview

This project implements a complete Machine Learning lifecycle to detect anomalous network behavior that may indicate potential data exfiltration attempts.

The system models normal network traffic behavior and detects deviations using unsupervised learning techniques.

An extended analysis includes:
- Unsupervised clustering evaluation  
- Label re-evaluation through consensus voting  
- Supervised model training  
- Comparative analysis between original and corrected labels  

---

## Problem Definition

The goal is to detect anomalous patterns in corporate network traffic that may indicate:

- Data exfiltration  
- Suspicious internal activity  
- Malicious behavior disguised as normal traffic  

### Key Characteristics

- Primarily unsupervised problem  
- No fully reliable labels in real-world scenarios  
- Attacks evolve over time  
- Normal behavior changes dynamically  

### Success Criteria

- Low false positive rate  
- Meaningful anomaly detection  
- Temporal stability  
- Practical usefulness for analysts  

---

## Dataset

**Dataset used:** NSL-KDD  

### Why NSL-KDD?

- Academic benchmark  
- 41 network features  
- Mixed numerical and categorical variables  
- Simulates normal and attack traffic  

### Limitations

- Synthetic dataset  
- No modern insider threat representation  
- Static distribution (no concept drift)  

---

## Data Exploration (EDA)

### Performed

- Dataset shape analysis  
- Label distribution analysis  
- Feature type inspection  
- Correlation matrix visualization  
- Variance analysis  

### Findings

- No missing values  
- Strong class imbalance  
- Highly correlated feature groups  
- Near-zero variance features  

---

## Data Quality Audit

### Removed Feature

- `num_outbound_cmds` (zero variance)

### Low Variance Features Identified

- land  
- urgent  
- num_failed_logins  
- root_shell  
- su_attempted  
- num_shells  
- num_access_files  
- is_host_login  
- is_guest_login  

Note: Not all were removed due to potential security relevance.

---

## Feature Reduction

Highly correlated feature groups were reduced using a hybrid manual-statistical approach.

### Removed Features

- num_outbound_cmds  
- num_root  
- srv_serror_rate  
- dst_host_srv_serror_rate  
- srv_rerror_rate  
- dst_host_srv_rerror_rate  

### Rationale

- Reduce redundancy  
- Improve model stability  
- Preserve representative signals  

---

## Preprocessing

Pipeline steps:

1. Filter training data (normal traffic only)  
2. Remove redundant features  
3. One-hot encode categorical variables:
   - protocol_type  
   - service  
   - flag  
4. Standardize numerical features (StandardScaler)  
5. Align test set features  

Key idea:

> Effective dimensionality depends on representation.

---

## Model Selection

**Model:** Isolation Forest  

### Reasons

- Unsupervised anomaly detection  
- Works in high-dimensional spaces  
- Robust to irrelevant features  
- Industry relevance  

### Training Strategy

- Train only on normal traffic  
- Detect anomalies in unseen data  

---

## Evaluation

Even though the approach is unsupervised, labels were used for evaluation.

### Metrics

- Precision  
- Recall  
- F1-score  
- Accuracy  

### Observations

- High precision  
- Moderate recall  
- Conservative anomaly threshold  

Important note:

Accuracy is not the main metric in anomaly detection.

---

## Extended Analysis

### Clustering Evaluation

Algorithms used:

- K-Means  
- Fuzzy C-Means  
- Subtractive Clustering  
- DBSCAN  
- Agglomerative (Ward)  

### Results

| Algorithm      | ARI    | NMI    | Silhouette |
|----------------|--------|--------|------------|
| K-Means        | 0.002  | 0.003  | 0.081      |
| Fuzzy C-Means  | 0.568  | 0.476  | 0.198      |
| Subtractive    | 0.562  | 0.546  | 0.214      |
| DBSCAN         | 0.042  | 0.150  | -0.072     |
| Agglomerative  | 0.411  | 0.344  | 0.194      |

Best performance: Fuzzy C-Means and Subtractive (ARI ≈ 0.56)

---

### Label Re-evaluation

- Consensus voting approach  
- Hungarian algorithm for alignment  
- 12,955 labels corrected (12.95%)  

Results:

- 868 normal → attack  
- 12,087 attack → normal  

---

### Supervised Models

| Model               | F1 (Original) | F1 (Relabeled) |
|---------------------|---------------|----------------|
| Decision Tree       | 99.1%         | 85.5%          |
| Logistic Regression | 95.8%         | 85.4%          |
| Linear Regression   | 94.8%         | 85.4%          |

---

### Comparative Analysis

- Original labels outperform corrected labels  
- Relabeling was too aggressive  
- Recall dropped significantly  

Conclusion:

Automated relabeling without expert validation can degrade performance.

---

## Project Structure
src/
├── data_loader.py
├── preprocessing.py
├── model.py
├── eda.py
├── data_quality.py
├── visualization.py
├── clustering.py
├── relabeling.py
├── supervised.py
├── comparison.py
├── main.py
└── main_extended.py

data/
├── KDDTrain+.txt
└── KDDTest+.txt

reports/figures/


---

## ML Lifecycle Covered

- Problem definition  
- Data acquisition  
- Data exploration  
- Data quality audit  
- Feature engineering  
- Unsupervised model training  
- Clustering evaluation  
- Label re-evaluation  
- Supervised training  
- Model comparison  
- Evaluation  

---

## Future Improvements

- Threshold calibration  
- Dynamic contamination tuning  
- PCA before clustering  
- Hyperparameter optimization  
- Concept drift simulation  
- Deployment pipeline  
- Model persistence  
- Real-time streaming  
- Confident Learning  
- Expert-in-the-loop validation  

---

## Key Takeaways

- Anomaly detection is threshold-sensitive  
- Feature redundancy affects stability  
- Low variance does not imply low importance  
- Representation defines effective dimensionality  
- ML supports analysts, not replaces them  
- Automatic relabeling is risky  
- High accuracy ≠ real-world performance  
- Clustering validates intrinsic structure  

---

## Generated Visualizations

Stored in:


reports/figures/


Includes:

- Label distribution  
- Feature distributions  
- Correlation matrix  
- Anomaly score distribution  
- PCA projections  
- Clustering comparisons  
- Confusion matrices  
- Model comparison charts  

---

## Final Reflection

A model that is not continuously updated will eventually classify the future as anomalous.

Continuous monitoring and retraining are essential in real-world cybersecurity systems.
