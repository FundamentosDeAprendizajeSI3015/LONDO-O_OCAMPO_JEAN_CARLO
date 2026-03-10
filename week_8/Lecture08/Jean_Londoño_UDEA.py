"""
FIRE-UdeA
Modelo Predictivo de Riesgo Financiero
Clasificación con Decision Tree, Random Forest y Gradient Boosting

Este script implementa un modelo de clasificación para predecir el riesgo financiero
basado en datos sintéticos. Utiliza algoritmos de aprendizaje automático como
Decision Tree, Random Forest y Gradient Boosting para clasificar instancias
en categorías de riesgo.

El flujo del script incluye:
- Carga y exploración inicial de datos (EDA)
- Limpieza de datos (imputación de valores faltantes)
- Entrenamiento de modelos con búsqueda de hiperparámetros
- Evaluación de modelos con métricas de clasificación
- Visualización de resultados (matrices de confusión, curvas ROC, importancia de características)

By: Jean Carlo Londoño Ocampo
Fecha: 10/03/2026
"""

# ==========================================================
# LIBRERÍAS
# ==========================================================

# Importaciones estándar de Python para manejo de datos y visualización
import pandas as pd  # Para manipulación de datos tabulares
import numpy as np   # Para operaciones numéricas y arrays
import matplotlib.pyplot as plt  # Para creación de gráficos
import seaborn as sns  # Para visualizaciones estadísticas avanzadas
import os  # Para operaciones del sistema de archivos

# Importaciones de scikit-learn para modelado y evaluación
from sklearn.model_selection import (
    train_test_split,      # Para dividir datos en entrenamiento y prueba
    cross_val_score,       # Para validación cruzada
    GridSearchCV,          # Para búsqueda de hiperparámetros
    StratifiedKFold        # Para validación cruzada estratificada
)

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # Para árboles de decisión
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Para ensembles

from sklearn.metrics import (
    accuracy_score,        # Precisión del modelo
    classification_report, # Reporte detallado de clasificación
    confusion_matrix,      # Matriz de confusión
    roc_auc_score,         # Área bajo la curva ROC
    roc_curve,             # Curva ROC
    f1_score,              # Puntaje F1
    precision_score,       # Precisión
    recall_score           # Recall (sensibilidad)
)

# Configuración para suprimir advertencias y mejorar visualizaciones
import warnings
warnings.filterwarnings("ignore")  # Ignorar advertencias para limpiar la salida

# Configuración de matplotlib para gráficos de mayor tamaño
plt.rcParams["figure.figsize"] = (10,6)
# Estilo de seaborn para gráficos más atractivos
sns.set_style("whitegrid")

# ==========================================================
# CARPETA PARA GUARDAR GRÁFICAS
# ==========================================================

# Directorio donde se guardarán todas las visualizaciones generadas
# Se crea el directorio si no existe, evitando errores si ya existe
OUTPUT_DIR = "graficas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(path):
    """
    Carga el dataset desde un archivo CSV y muestra información básica.

    Parámetros:
    path (str): Ruta al archivo CSV que contiene el dataset.

    Retorna:
    pd.DataFrame: DataFrame con los datos cargados.
    """
    print("\n===== LOAD DATA =====")

    # Leer el archivo CSV usando pandas
    df = pd.read_csv(path)

    # Mostrar dimensiones del dataset (filas, columnas)
    print(f"Dataset cargado: {df.shape}")
    # Mostrar las primeras 5 filas para inspección visual
    print(df.head())

    return df

# ==========================================================
# BASIC EDA
# ==========================================================

def basic_eda(df):
    """
    Realiza un Análisis Exploratorio de Datos (EDA) básico.

    Incluye información del dataset, estadísticas descriptivas,
    valores faltantes y distribución de la variable objetivo.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos a analizar.
    """
    print("\n===== DATASET INFO =====")
    # Mostrar información general: tipos de datos, valores no nulos, etc.
    print(df.info())

    print("\n===== DESCRIPTIVE STATS =====")
    # Estadísticas descriptivas para variables numéricas
    print(df.describe())

    print("\n===== MISSING VALUES =====")
    # Conteo de valores faltantes por columna
    print(df.isnull().sum())

    print("\n===== CLASS DISTRIBUTION =====")
    # Distribución de la variable objetivo (label)
    print(df["label"].value_counts())

    # Visualizar la distribución de la variable objetivo
    sns.countplot(x="label", data=df)
    plt.title("Distribución Variable Objetivo")
    # Guardar el gráfico como imagen PNG
    plt.savefig(f"{OUTPUT_DIR}/distribucion_label.png", dpi=300, bbox_inches="tight")
    plt.close()  # Cerrar la figura para liberar memoria

# ==========================================================
# HISTOGRAMS
# ==========================================================

def plot_histograms(df, features):
    """
    Genera histogramas superpuestos para cada característica,
    diferenciando por clase (label 0 y 1).

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    features (list): Lista de nombres de columnas a graficar.
    """
    # Crear una cuadrícula de subplots 4x4 para múltiples histogramas
    fig, axes = plt.subplots(4,4, figsize=(16,14))
    axes = axes.flatten()  # Aplanar el array de ejes para iterar fácilmente

    # Iterar sobre cada característica
    for i, col in enumerate(features):
        ax = axes[i]

        # Histograma para clase 0 (verde)
        df[df["label"]==0][col].hist(
            bins=15,        # Número de bins
            alpha=0.6,      # Transparencia
            color="#2ecc71", # Color verde
            ax=ax
        )

        # Histograma para clase 1 (rojo), superpuesto
        df[df["label"]==1][col].hist(
            bins=15,
            alpha=0.6,
            color="#e74c3c", # Color rojo
            ax=ax
        )

        ax.set_title(col)  # Título del subplot con el nombre de la columna

    plt.tight_layout()  # Ajustar el layout para evitar solapamientos

    # Guardar la figura
    plt.savefig(f"{OUTPUT_DIR}/histogramas_variables.png", dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

def correlation_matrix(df, features):
    """
    Calcula y visualiza la matriz de correlación entre características y la variable objetivo.

    Imputa valores faltantes con la mediana antes del cálculo.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    features (list): Lista de características numéricas.
    """
    # Seleccionar solo columnas numéricas y la etiqueta
    df_numeric = df[features + ["label"]].copy()

    # Imputar valores faltantes con la mediana para cada característica
    for col in features:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

    # Calcular la matriz de correlación
    corr = df_numeric.corr()

    # Visualizar la matriz de correlación con un heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(
        corr,
        annot=True,        # Mostrar valores en las celdas
        fmt=".2f",         # Formato de 2 decimales
        cmap="RdBu_r",     # Mapa de colores divergente
        center=0           # Centro del mapa en 0
    )

    plt.title("Matriz de Correlación")

    # Guardar la figura
    plt.savefig(f"{OUTPUT_DIR}/matriz_correlacion.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Mostrar correlaciones con la variable objetivo, ordenadas
    print("\nCorrelación con label:")
    print(corr["label"].sort_values(ascending=False))

# ==========================================================
# DATA CLEANING
# ==========================================================

def clean_data(df):
    """
    Limpia el dataset imputando valores faltantes con la mediana.

    Excluye columnas no predictoras como 'anio', 'unidad' y 'label'.

    Parámetros:
    df (pd.DataFrame): DataFrame original.

    Retorna:
    tuple: (df_limpio, lista_características)
    """
    print("\n===== CLEANING DATA =====")

    df_model = df.copy()  # Copia para no modificar el original

    # Seleccionar columnas predictoras (excluir no predictoras)
    feature_cols = [
        c for c in df.columns
        if c not in ["anio","unidad","label"]
    ]

    # Imputar valores faltantes con la mediana para cada característica
    for col in feature_cols:
        n_null = df_model[col].isnull().sum()

        if n_null > 0:
            median_val = df_model[col].median()
            df_model[col] = df_model[col].fillna(median_val)
            print(f"{col} → {n_null} nulos imputados con mediana")

    return df_model, feature_cols

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_data(df_model, features):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.

    Utiliza estratificación para mantener la proporción de clases.

    Parámetros:
    df_model (pd.DataFrame): DataFrame limpio.
    features (list): Lista de características.

    Retorna:
    tuple: (X_train, X_test, y_train, y_test)
    """
    # Separar características (X) y variable objetivo (y)
    X = df_model[features]
    y = df_model["label"]

    # Dividir en train/test con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,      # 25% para prueba
        stratify=y,          # Mantener proporción de clases
        random_state=42      # Para reproducibilidad
    )

    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)

    return X_train, X_test, y_train, y_test

# ==========================================================
# DECISION TREE
# ==========================================================

def train_decision_tree(X_train, y_train):
    """
    Entrena un modelo de Decision Tree con búsqueda de hiperparámetros.

    Utiliza GridSearchCV con validación cruzada estratificada.

    Parámetros:
    X_train (pd.DataFrame): Características de entrenamiento.
    y_train (pd.Series): Variable objetivo de entrenamiento.

    Retorna:
    sklearn model: Mejor modelo entrenado.
    """
    print("\n===== TRAIN DECISION TREE =====")

    # Definir el espacio de hiperparámetros a explorar
    params = {
        "max_depth": [2,3,4,5,6,None],        # Profundidad máxima del árbol
        "min_samples_split": [2,5,10],        # Mínimas muestras para dividir
        "min_samples_leaf": [1,3,5],          # Mínimas muestras en hoja
        "criterion": ["gini","entropy"]       # Criterio de división
    }

    # Validación cruzada estratificada para mantener proporción de clases
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Búsqueda en cuadrícula con métrica F1
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),  # Modelo base
        params,                                   # Espacio de parámetros
        cv=cv,                                    # Validación cruzada
        scoring="f1",                             # Métrica de evaluación
        n_jobs=-1                                 # Usar todos los núcleos
    )

    # Entrenar el modelo
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)

    return grid.best_estimator_  # Devolver el mejor modelo

# ==========================================================
# RANDOM FOREST
# ==========================================================

def train_random_forest(X_train, y_train):
    """
    Entrena un modelo de Random Forest con búsqueda de hiperparámetros.

    Utiliza GridSearchCV con validación cruzada estratificada.

    Parámetros:
    X_train (pd.DataFrame): Características de entrenamiento.
    y_train (pd.Series): Variable objetivo de entrenamiento.

    Retorna:
    sklearn model: Mejor modelo entrenado.
    """
    print("\n===== TRAIN RANDOM FOREST =====")

    # Definir el espacio de hiperparámetros
    params = {
        "n_estimators": [50,100,200],         # Número de árboles
        "max_depth": [3,5,7,None],            # Profundidad máxima
        "min_samples_split": [2,5,10],        # Mínimas muestras para dividir
        "min_samples_leaf": [1,3,5],          # Mínimas muestras en hoja
        "max_features": ["sqrt","log2"]       # Número de características a considerar
    }

    # Validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Búsqueda en cuadrícula
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        params,
        cv=cv,
        scoring="f1",    # Optimizar para F1-score
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)

    return grid.best_estimator_

# ==========================================================
# MODEL EVALUATION
# ==========================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """
    Evalúa un modelo entrenado en los conjuntos de train y test.

    Calcula métricas de clasificación y genera matriz de confusión.

    Parámetros:
    model: Modelo entrenado de sklearn.
    X_train, X_test: Características de train/test.
    y_train, y_test: Etiquetas de train/test.
    name (str): Nombre del modelo para reportes.

    Retorna:
    np.array: Probabilidades de la clase positiva para test.
    """
    print(f"\n===== {name} =====")

    # Predicciones en train y test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilidades para ROC-AUC (solo clase positiva)
    y_test_proba = model.predict_proba(X_test)[:,1]

    # Calcular y mostrar métricas
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall:", recall_score(y_test, y_test_pred))
    print("F1:", f1_score(y_test, y_test_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_test_proba))

    # Reporte de clasificación detallado
    print("\nClassification Report")
    print(classification_report(y_test, y_test_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt="d")  # Mostrar valores como enteros
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return y_test_proba

# ==========================================================
# ROC CURVE
# ==========================================================

def plot_roc(models, X_test, y_test):
    """
    Genera y guarda una gráfica de curvas ROC para múltiples modelos.

    Parámetros:
    models (dict): Diccionario con nombres de modelos como keys y modelos como values.
    X_test (pd.DataFrame): Características de test.
    y_test (pd.Series): Etiquetas de test.
    """
    plt.figure()

    # Iterar sobre cada modelo
    for name, model in models.items():
        # Obtener probabilidades de la clase positiva
        y_proba = model.predict_proba(X_test)[:,1]

        # Calcular FPR, TPR y AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        # Graficar la curva ROC
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # Línea diagonal de referencia (clasificador aleatorio)
    plt.plot([0,1], [0,1], "--")

    plt.legend()
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")

    # Guardar la gráfica
    plt.savefig(f"{OUTPUT_DIR}/curvas_roc_modelos.png", dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance(model, features, title):
    """
    Visualiza la importancia de las características del modelo.

    Parámetros:
    model: Modelo con atributo feature_importances_ (ej. Random Forest).
    features (list): Lista de nombres de características.
    title (str): Título para la gráfica.
    """
    # Crear serie con importancias ordenadas
    imp = pd.Series(
        model.feature_importances_,  # Importancias del modelo
        index=features               # Nombres de características
    ).sort_values()  # Ordenar de menor a mayor

    # Graficar como barras horizontales
    imp.plot(kind="barh")
    plt.title(title)

    # Guardar la gráfica
    plt.savefig(f"{OUTPUT_DIR}/feature_importance_random_forest.png", dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================================
# MAIN
# ==========================================================

def main():
    """
    Función principal que ejecuta todo el pipeline del modelo.

    Incluye carga de datos, EDA, limpieza, entrenamiento y evaluación.
    """
    # Cargar datos
    df = load_data("dataset_sintetico_FIRE_UdeA_realista.csv")

    # Análisis exploratorio básico
    basic_eda(df)

    # Seleccionar características (excluir no predictoras)
    features = [
        c for c in df.columns
        if c not in ["anio","unidad","label"]
    ]

    # Visualizaciones
    plot_histograms(df, features)
    correlation_matrix(df, features)

    # Limpieza de datos
    df_model, features = clean_data(df)

    # División train/test
    X_train, X_test, y_train, y_test = split_data(df_model, features)

    # Entrenamiento de modelos
    dt = train_decision_tree(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    # Gradient Boosting con parámetros fijos (no optimizado)
    gb = GradientBoostingClassifier(
        n_estimators=100,     # Número de estimadores
        max_depth=3,          # Profundidad máxima
        learning_rate=0.1,    # Tasa de aprendizaje
        random_state=42       # Reproducibilidad
    )
    gb.fit(X_train, y_train)  # Entrenar el modelo

    # Evaluación de modelos
    evaluate_model(dt, X_train, X_test, y_train, y_test, "Decision_Tree")
    evaluate_model(rf, X_train, X_test, y_train, y_test, "Random_Forest")
    evaluate_model(gb, X_train, X_test, y_train, y_test, "Gradient_Boosting")

    # Comparación de curvas ROC
    plot_roc({
        "Decision Tree": dt,
        "Random Forest": rf,
        "Gradient Boosting": gb
    }, X_test, y_test)

    # Importancia de características (solo para Random Forest)
    feature_importance(rf, features, "Random Forest Feature Importance")

# Punto de entrada del script
if __name__ == "__main__":
    main()