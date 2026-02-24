"""
Jean Carlo Londoño Ocampo 
Supervised Trees Exercise
Random Forest vs Gradient Boosting
NSL-KDD Dataset (Merged Version)

Week 6 – Decision Trees & Ensembles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ==============================
# 1. LOAD DATA
# ==============================

TRAIN_PATH = r"C:\Users\USUARIO\Desktop\repoml\LONDO-O_OCAMPO_JEAN_CARLO\Insight_X_Project\data\KDDTrain+.txt"
TEST_PATH  = r"C:\Users\USUARIO\Desktop\repoml\LONDO-O_OCAMPO_JEAN_CARLO\Insight_X_Project\data\KDDTest+.txt"

COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count",
    "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

train_df = pd.read_csv(TRAIN_PATH, names=COLUMNS)
test_df  = pd.read_csv(TEST_PATH, names=COLUMNS)

# Merge and shuffle
df = pd.concat([train_df, test_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", df.shape)

# ==============================
# 2. BASIC EDA
# ==============================

print("\nClass distribution:")
print(df["label"].value_counts())

sns.countplot(x="label", data=df)
plt.xticks(rotation=90)
plt.title("Class Distribution")
plt.show()

# ==============================
# 3. CLEANING & TRANSFORMATION
# ==============================

# Convert to binary classification
df["target"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

# Drop unused columns
df = df.drop(columns=["label", "difficulty"])

# One-hot encoding for categorical features
df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

# ==============================
# 4. DEFINE FEATURES & TARGET
# ==============================

X = df.drop(columns=["target"])
y = df["target"]

print("Feature matrix shape:", X.shape)

# ==============================
# 5. EXTREME REDUCTION EXPERIMENT (5% TRAIN)
# ==============================

# First: create small training subset (5% of full dataset)
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y,
    test_size=0.95,
    random_state=42,
    stratify=y
)

print("Training size (5% approx):", X_train.shape)

# Now split remaining data into validation and test (each 50% of remaining)
X_val, X_test, y_val, y_test = train_test_split(
    X_remaining, y_remaining,
    test_size=0.5,
    random_state=42,
    stratify=y_remaining
)

print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)

# ==============================
# 6. SCALING
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ==============================
# 7. RANDOM FOREST
# ==============================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

rf_val_pred  = rf_model.predict(X_val_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

# ==============================
# 8. GRADIENT BOOSTING
# ==============================

gb_model = GradientBoostingClassifier(
    n_estimators=100,  # Reduced for faster training
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)

gb_val_pred  = gb_model.predict(X_val_scaled)
gb_test_pred = gb_model.predict(X_test_scaled)

# ==============================
# 9. EVALUATION FUNCTION
# ==============================

def evaluate_model(name, dataset_name, y_true, y_pred):
    print(f"\n=== {name} - {dataset_name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - {dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==============================
# 10. EVALUATION RESULTS
# ==============================

# Random Forest
evaluate_model("Random Forest", "Validation", y_val, rf_val_pred)
evaluate_model("Random Forest", "Test", y_test, rf_test_pred)

# Gradient Boosting
evaluate_model("Gradient Boosting", "Validation", y_val, gb_val_pred)
evaluate_model("Gradient Boosting", "Test", y_test, gb_test_pred)

importances = rf_model.feature_importances_
feature_names = X.columns

feat_imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# show top 15
top_n = 15
plt.figure(figsize=(10,6))
sns.barplot(
    x="importance",
    y="feature",
    data=feat_imp_df.head(top_n)
)
plt.title("Top 15 Feature Importances - Random Forest")
plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(
    rf_model.estimators_[0],   # Plot the first tree in the forest
    feature_names=X.columns,
    class_names=["Normal", "Attack"],
    filled=True,
    max_depth=3  # Limit depth for better visualization
)
plt.show()