"""
=============================================================
COMENTADO Y REVISADO POR: Jean Carlo Londoño Ocampo 
Laboratorio: Preprocesamiento y exploración inicial
Dataset: Iris (scikit-learn)

Autor: Jean Carlo Londoño Ocampo 

Objetivo:
Construir una representación X, y de buena calidad
antes de entrenar cualquier modelo.

Este laboratorio cubre:
1. Carga del dataset
2. Exploración inicial (EDA)
3. Chequeos de calidad de datos
4. Detección básica de outliers
5. Escalado (estandarización)
6. Visualización de patrones predominantes
7. Partición train / test
8. Exportación de datasets
=============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from babyplots import Babyplot

# ============================================================
# 1. CARGA DEL DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Carga el dataset Iris y construye el DataFrame.

    En términos de ML:
    aquí estamos construyendo la representación inicial (X, y)
    a partir de la fuente de datos.
    """
    
    print("Cargando dataset Iris...")

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["target"] = iris.target

    return df

# ============================================================
# 2. VISUALIZACIÓN 3D CON BABYPLOTS
# ============================================================

def plot_3d_babyplots(df: pd.DataFrame):
    """
    Visualización exploratoria 3D.

    Esta visualización no es para entrenar modelos,
    sino para observar si existen patrones predominantes
    (clusters naturales) en la representación X (los que nos enseñó la profe).
    """

    print("Generando visualización 3D (Babyplots)...")

    bp = Babyplot()

    data_points = df[
        ["sepal length (cm)",
         "sepal width (cm)",
         "petal length (cm)"]
    ].values

    colors = df["target"].values

    bp.add_plot(
        data_points,
        "pointCloud",
        "categories",
        colors,
        {"colorScale": "Set2"}
    )

    print(bp)

# ============================================================
# 3. EXPLORACIÓN INICIAL Y CHEQUEOS DE CALIDAD
# ============================================================

def basic_eda(df: pd.DataFrame):
    """
    Exploración básica.

    Relación con calidad de datos:
    - completitud
    - validez
    - unicidad (parcial)
    """
    print("\nInformación general:")
    print(df.info())

    print("\nEstadísticas descriptivas:")
    print(df.describe())

    print("\nValores únicos por columna:")
    print(df.nunique())

    print("\nChequeo de valores nulos (completitud):")
    print(df.isna().sum())

# ============================================================
# 4. DETECCIÓN BÁSICA DE OUTLIERS
# ============================================================

def detect_outliers(df: pd.DataFrame) -> pd.Series:
    """
    Detección simple de outliers usando z-score.

    Conceptualmente:
    buscamos puntos con baja probabilidad
    bajo la distribución empírica de X.

    Esto conecta con:
    detección de anomalías y calidad de representación.
    """

    numeric_data = df.select_dtypes(include=[np.number])

    z_scores = np.abs(stats.zscore(numeric_data))

    outlier_mask = (z_scores > 3).any(axis=1)

    print(
        f"Posibles outliers detectados: {outlier_mask.sum()}"
    )

    return outlier_mask

# ============================================================
# 5. VISUALIZACIONES ADICIONALES (EDA GRÁFICO)
# ============================================================

def extra_visualizations(df: pd.DataFrame):
    """
    Visualizaciones clásicas para entender patrones predominantes.

    Estas gráficas ayudan a responder:
    ¿cómo está distribuido X?
    ¿existen separaciones naturales entre clases?
    """

    print("\nGenerando visualizaciones adicionales...")

    # -------------------------------
    # Pairplot (relaciones bivariadas)
    # -------------------------------
    sns.pairplot(
        df,
        hue="target",
        diag_kind="kde"
    )
    plt.suptitle("Pairplot - Relaciones entre variables", y=1.02)
    plt.show()

    # -------------------------------
    # Mapa de correlaciones
    # -------------------------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df.drop(columns="target").corr(),
        annot=True,
        cmap="coolwarm"
    )
    plt.title("Matriz de correlación")
    plt.show()

# ============================================================
# 6. ESCALADO Y CONSTRUCCIÓN DE X, y
# ============================================================

def build_scaled_features(df: pd.DataFrame):
    """
    Construcción formal de X e y.

    Aquí aplicamos estandarización porque:
    - los modelos que veremos (regresión, SVM, PCA, redes)
      son sensibles a escala.
    - queremos una representación X centrada y comparable.

    Esto es una transformación de representación:
        X  ->  X'
    """

    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()

    # MUY IMPORTANTE:
    # en un pipeline real el fit debe hacerse SOLO con train.
    # Aquí lo hacemos antes solo por simplicidad didáctica.
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    print("\nEjemplo de X estandarizado:")
    print(X_scaled.head())

    return X_scaled, y

# ============================================================
# 7. VISUALIZACIÓN DE PATRONES DOMINANTES CON PCA
# ============================================================

def pca_visualization(X: pd.DataFrame, y: pd.Series):
    """
    PCA no es un clasificador.

    Es una técnica para descubrir direcciones de máxima varianza.

    Conecta directamente con:
    detección de patrones predominantes.
    """

    pca = PCA(n_components=2)

    Z = pca.fit_transform(X)

    df_pca = pd.DataFrame(Z, columns=["PC1", "PC2"])
    df_pca["target"] = y.values

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="target",
        palette="Set2"
    )

    plt.title("PCA (2 componentes) – patrones predominantes")
    plt.show()

    print("Varianza explicada por cada componente:")
    print(pca.explained_variance_ratio_)

# ============================================================
# 8. PARTICIÓN TRAIN / TEST
# ============================================================

def split_data(X, y):
    """
    Separación train/test.

    Es un control de calidad experimental:
    evitamos evaluar el modelo sobre los mismos datos
    usados para construir la función.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTamaños de conjuntos:")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    return X_train, X_test, y_train, y_test

# ============================================================
# 9. EXPORTACIÓN
# ============================================================

def export_datasets(X_train, X_test, y_train, y_test):
    """
    Exporta los datasets listos para modelado.

    Buena práctica:
    el preprocesamiento queda desacoplado
    del entrenamiento del modelo.
    """
    
    output_dir = Path("data_output")
    output_dir.mkdir(exist_ok=True)

    train_path = output_dir / "iris_train.parquet"
    test_path = output_dir / "iris_test.parquet"

    X_train.assign(target=y_train).to_parquet(
        train_path,
        index=False
    )

    X_test.assign(target=y_test).to_parquet(
        test_path,
        index=False
    )

    print("\nArchivos exportados:")
    print(train_path)
    print(test_path)


# ============================================================
# 10. PIPELINE PRINCIPAL
# ============================================================

def main():
    """
    Pipeline de preparación de datos.

    Analógicamente:
    es como preparar el terreno antes de construir una casa.
    El modelo es la casa.
    X, y es el terreno.
    """

    df = load_dataset()

    print("\nPrimeras filas del dataset:")
    print(df.head())

    plot_3d_babyplots(df)

    basic_eda(df)

    outlier_mask = detect_outliers(df)

    # Por defecto no eliminamos outliers
    # porque Iris es un dataset limpio y pequeño.
    # df = df[~outlier_mask]

    extra_visualizations(df)

    X_scaled, y = build_scaled_features(df)

    pca_visualization(X_scaled, y)

    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y
    )

    export_datasets(
        X_train, X_test,
        y_train, y_test
    )

    print("\nLaboratorio finalizado correctamente.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()