# Network Anomaly Detection using Isolation Forest  
NSL-KDD Dataset – Machine Learning Lifecycle Project  - Insight X

Author: Jean Carlo Londoño Ocampo  
Course: Fundamentos de Aprendizaje Automático  
Year: 2026  

---

## Project Overview

This project implements a complete Machine Learning lifecycle to detect anomalous network behavior that may indicate potential data exfiltration attempts.

The system models normal network traffic behavior and detects deviations using unsupervised learning techniques.
---

## Problem Definition

The goal is to detect anomalous patterns in corporate network traffic that may indicate:

- Data exfiltration
- Suspicious internal activity
- Malicious behavior disguised as normal traffic

### Key Characteristics:

- Primarily unsupervised problem
- No fully reliable attack labels in real-world scenarios
- Attacks evolve over time
- Normal behavior changes dynamically

Success is defined as:

- Low false positive rate
- Meaningful anomaly detection
- Stability over time
- Operational usefulness for security analysts

---

## Dataset

Dataset used: **NSL-KDD**

Why NSL-KDD?

- Academic benchmark
- Contains 41 network features
- Mix of numerical and categorical variables
- Simulates normal and attack traffic

Limitations:

- Synthetic dataset
- Does not represent modern insider threats
- Static distribution (no concept drift)
---

## Data Exploration (EDA)

Performed:

- Dataset shape analysis
- Label distribution analysis
- Feature type inspection
- Correlation matrix visualization
- Variance analysis

Findings:

- No missing values
- Strong class imbalance
- Presence of highly correlated feature groups
- Some zero and near-zero variance features

---

## Data Quality Audit

### Zero Variance Feature Removed:

- `num_outbound_cmds`

### Low Variance Features Identified:

- land
- urgent
- num_failed_logins
- root_shell
- su_attempted
- num_shells
- num_access_files
- is_host_login
- is_guest_login

(Note: Not all low variance features were removed due to potential security relevance.)

---

## Correlation-Based Feature Reduction

Highly correlated feature groups were identified.

Manual + statistical hybrid selection approach was used.

Removed features:

- num_outbound_cmds
- num_root
- srv_serror_rate
- dst_host_srv_serror_rate
- srv_rerror_rate
- dst_host_srv_rerror_rate

Rationale:

- Reduce redundancy
- Improve model stability
- Preserve representative signals

---

## Preprocessing Steps

1. Filter training data to include only normal traffic
2. Remove selected redundant features
3. One-hot encode categorical variables:
   - protocol_type
   - service
   - flag
4. Standardize numerical features using StandardScaler
5. Align test set features with training set columns

Important concept:

> Effective dimensionality depends on representation.

---

## Model Selection

Model: Isolation Forest

Reasons:

- Unsupervised anomaly detection
- Scales to high dimensional data
- Robust to irrelevant features
- Widely used in industry

Training Strategy:

- Train only on normal traffic
- Detect deviations in test data

---

##  Evaluation

Even though the approach is unsupervised, labels were used for academic evaluation.

Metrics used:

- Precision
- Recall
- F1-score
- Accuracy

Observations:

- High precision in attack detection
- Moderate recall
- Conservative anomaly threshold

Note:

Accuracy is not the primary metric in anomaly detection.

---

## Project Structure

src/
│
├── data_loader.py
├── preprocessing.py
├── model.py
├── eda.py
├── data_quality.py
└── main.py

data/
│
├── KDDTrain+.txt
└── KDDTest+.txt


---

## 🔄 ML Lifecycle Covered

✔ Problem definition  
✔ Data acquisition  
✔ Data exploration  
✔ Data quality audit  
✔ Feature engineering  
✔ Model training  
✔ Evaluation  

---

## Future Improvements

- Threshold calibration
- Dynamic contamination tuning
- PCA dimensionality reduction
- Hyperparameter optimization
- Concept drift simulation
- Deployment pipeline simulation
- Model persistence (joblib)
- Real-time streaming simulation

---

## Important Conceptual Takeaways

- Anomaly detection is threshold-sensitive
- Feature redundancy affects model stability
- Low variance does not always mean low importance
- Effective dimensionality depends on representation
- Machine Learning supports analysts, it does not replace them

---

## Final Reflection

A model that is not continuously updated will eventually classify the future as anomalous.

Continuous monitoring and retraining are essential in real-world cybersecurity systems.