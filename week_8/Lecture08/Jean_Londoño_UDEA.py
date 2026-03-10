"""
FIRE-UdeA
Modelo Predictivo de Riesgo Financiero
Clasificación con Decision Tree, Random Forest y Gradient Boosting
Universidad de Antioquia
By: Jean Carlo Londoño Ocampo
"""

# ==========================================================
# LIBRERÍAS
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (10,6)
sns.set_style("whitegrid")

# ==========================================================
# CARPETA PARA GUARDAR GRÁFICAS
# ==========================================================

OUTPUT_DIR = "graficas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(path):

    print("\n===== LOAD DATA =====")

    df = pd.read_csv(path)

    print(f"Dataset cargado: {df.shape}")
    print(df.head())

    return df


# ==========================================================
# BASIC EDA
# ==========================================================

def basic_eda(df):

    print("\n===== DATASET INFO =====")
    print(df.info())

    print("\n===== DESCRIPTIVE STATS =====")
    print(df.describe())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== CLASS DISTRIBUTION =====")
    print(df["label"].value_counts())

    sns.countplot(x="label", data=df)
    plt.title("Distribución Variable Objetivo")

    plt.savefig(f"{OUTPUT_DIR}/distribucion_label.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================================
# HISTOGRAMS
# ==========================================================

def plot_histograms(df, features):

    fig, axes = plt.subplots(4,4, figsize=(16,14))
    axes = axes.flatten()

    for i,col in enumerate(features):

        ax = axes[i]

        df[df["label"]==0][col].hist(
            bins=15,
            alpha=0.6,
            color="#2ecc71",
            ax=ax
        )

        df[df["label"]==1][col].hist(
            bins=15,
            alpha=0.6,
            color="#e74c3c",
            ax=ax
        )

        ax.set_title(col)

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/histogramas_variables.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================================
# CORRELATION MATRIX
# ==========================================================

def correlation_matrix(df, features):

    df_numeric = df[features + ["label"]].copy()

    for col in features:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

    corr = df_numeric.corr()

    plt.figure(figsize=(12,10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0
    )

    plt.title("Matriz de Correlación")

    plt.savefig(f"{OUTPUT_DIR}/matriz_correlacion.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nCorrelación con label:")
    print(corr["label"].sort_values(ascending=False))


# ==========================================================
# DATA CLEANING
# ==========================================================

def clean_data(df):

    print("\n===== CLEANING DATA =====")

    df_model = df.copy()

    feature_cols = [
        c for c in df.columns
        if c not in ["anio","unidad","label"]
    ]

    for col in feature_cols:

        n_null = df_model[col].isnull().sum()

        if n_null > 0:

            median_val = df_model[col].median()

            df_model[col] = df_model[col].fillna(median_val)

            print(
                f"{col} → {n_null} nulos imputados con mediana"
            )

    return df_model, feature_cols


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_data(df_model, features):

    X = df_model[features]
    y = df_model["label"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ==========================================================
# DECISION TREE
# ==========================================================

def train_decision_tree(X_train, y_train):

    print("\n===== TRAIN DECISION TREE =====")

    params = {

        "max_depth":[2,3,4,5,6,None],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,3,5],
        "criterion":["gini","entropy"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(

        DecisionTreeClassifier(random_state=42),

        params,

        cv=cv,

        scoring="f1",

        n_jobs=-1
    )

    grid.fit(X_train,y_train)

    print("Best params:", grid.best_params_)

    return grid.best_estimator_


# ==========================================================
# RANDOM FOREST
# ==========================================================

def train_random_forest(X_train, y_train):

    print("\n===== TRAIN RANDOM FOREST =====")

    params = {

        "n_estimators":[50,100,200],
        "max_depth":[3,5,7,None],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,3,5],
        "max_features":["sqrt","log2"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(

        RandomForestClassifier(random_state=42),

        params,

        cv=cv,

        scoring="f1",

        n_jobs=-1
    )

    grid.fit(X_train,y_train)

    print("Best params:", grid.best_params_)

    return grid.best_estimator_


# ==========================================================
# MODEL EVALUATION
# ==========================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, name):

    print(f"\n===== {name} =====")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_test_proba = model.predict_proba(X_test)[:,1]

    print("Train Accuracy:", accuracy_score(y_train,y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test,y_test_pred))

    print("Precision:", precision_score(y_test,y_test_pred))
    print("Recall:", recall_score(y_test,y_test_pred))
    print("F1:", f1_score(y_test,y_test_pred))

    print("ROC-AUC:", roc_auc_score(y_test,y_test_proba))

    print("\nClassification Report")
    print(classification_report(y_test,y_test_pred))

    cm = confusion_matrix(y_test,y_test_pred)

    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - {name}")

    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return y_test_proba


# ==========================================================
# ROC CURVE
# ==========================================================

def plot_roc(models, X_test, y_test):

    plt.figure()

    for name,model in models.items():

        y_proba = model.predict_proba(X_test)[:,1]

        fpr,tpr,_ = roc_curve(y_test,y_proba)

        auc = roc_auc_score(y_test,y_proba)

        plt.plot(fpr,tpr,label=f"{name} (AUC={auc:.3f})")

    plt.plot([0,1],[0,1],"--")

    plt.legend()
    plt.title("ROC Curves")
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.savefig(f"{OUTPUT_DIR}/curvas_roc_modelos.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance(model,features,title):

    imp = pd.Series(

        model.feature_importances_,

        index=features

    ).sort_values()

    imp.plot(kind="barh")

    plt.title(title)

    plt.savefig(f"{OUTPUT_DIR}/feature_importance_random_forest.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():

    df = load_data(
        "dataset_sintetico_FIRE_UdeA_realista.csv"
    )

    basic_eda(df)

    features = [
        c for c in df.columns
        if c not in ["anio","unidad","label"]
    ]

    plot_histograms(df,features)

    correlation_matrix(df,features)

    df_model,features = clean_data(df)

    X_train,X_test,y_train,y_test = split_data(
        df_model,
        features
    )

    dt = train_decision_tree(X_train,y_train)

    rf = train_random_forest(X_train,y_train)

    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    gb.fit(X_train,y_train)

    evaluate_model(
        dt,X_train,X_test,y_train,y_test,"Decision_Tree"
    )

    evaluate_model(
        rf,X_train,X_test,y_train,y_test,"Random_Forest"
    )

    evaluate_model(
        gb,X_train,X_test,y_train,y_test,"Gradient_Boosting"
    )

    plot_roc(

        {
            "Decision Tree":dt,
            "Random Forest":rf,
            "Gradient Boosting":gb
        },

        X_test,
        y_test
    )

    feature_importance(
        rf,
        features,
        "Random Forest Feature Importance"
    )


if __name__ == "__main__":
    main()