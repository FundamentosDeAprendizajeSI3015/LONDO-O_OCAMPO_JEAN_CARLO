"""
Exploración, limpieza y transformación de datos para movies.csv
Autor: Jean Londoño

Curso: Fundamentos de Machine Learning (6° semestre)

Objetivo:
Construir un flujo básico de preprocesamiento de datos que incluya:
- limpieza
- estadística descriptiva
- detección de outliers
- visualización
- codificación de variables categóricas
- escalado
- transformaciones para modelos de ML
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# -------------------------------------------------------------
# Lectura del dataset
# -------------------------------------------------------------
def read_movies(path):
    """
    Lee el archivo movies.csv utilizando el motor de Python,
    ya que existen campos con comas internas.
    """
    return pd.read_csv(path, engine="python", skipinitialspace=True)

# -------------------------------------------------------------
# Limpieza y normalización de columnas
# -------------------------------------------------------------
def clean_columns(df):
    """
    Limpia nombres de columnas, convierte tipos y estandariza
    valores faltantes.
    """
    df = df.copy()

    # Normalización de nombres
    df.columns = [c.strip() for c in df.columns]

    # Limpieza general de columnas tipo texto
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan})

    # Conversión explícita de columnas numéricas
    if "RATING" in df.columns:
        df["RATING"] = pd.to_numeric(df["RATING"], errors="coerce")

    if "VOTES" in df.columns:
        df["VOTES"] = (
            df["VOTES"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df["VOTES"] = pd.to_numeric(df["VOTES"], errors="coerce")

    if "RunTime" in df.columns:
        df["RunTime"] = pd.to_numeric(df["RunTime"], errors="coerce")

    if "Gross" in df.columns:
        df["Gross"] = (
            df["Gross"]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
        )
        df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")

    return df

# -------------------------------------------------------------
# Medidas de tendencia central
# -------------------------------------------------------------
def central_tendency(df, cols):
    """
    Calcula media, mediana y moda para columnas numéricas.
    """
    results = {}

    for c in cols:
        s = df[c].dropna()

        if s.empty:
            results[c] = None
            continue

        results[c] = {
            "mean": s.mean(),
            "median": s.median(),
            "mode": s.mode().tolist()
        }

    return results

# -------------------------------------------------------------
# Medidas de dispersión
# -------------------------------------------------------------
def dispersion_measures(df, cols):
    """
    Calcula desviación estándar, varianza, rango e IQR.
    """
    results = {}

    for c in cols:
        s = df[c].dropna()

        if s.empty:
            results[c] = None
            continue

        results[c] = {
            "std": s.std(),
            "var": s.var(),
            "range": s.max() - s.min(),
            "iqr": s.quantile(0.75) - s.quantile(0.25)
        }

    return results

# -------------------------------------------------------------
# Medidas de posición
# -------------------------------------------------------------
def position_measures(df, cols):
    """
    Calcula mínimos, máximos y cuartiles.
    """
    results = {}

    for c in cols:
        s = df[c].dropna()

        if s.empty:
            results[c] = None
            continue

        results[c] = {
            "min": s.min(),
            "q1": s.quantile(0.25),
            "median": s.quantile(0.50),
            "q3": s.quantile(0.75),
            "max": s.max()
        }

    return results

# -------------------------------------------------------------
# Eliminación de outliers usando IQR
# -------------------------------------------------------------
def remove_outliers_iqr(df, cols, factor=1.5):
    """
    Elimina filas que presentan valores atípicos en
    cualquiera de las columnas indicadas.
    """
    df = df.copy()
    mask = pd.Series(True, index=df.index)

    for c in cols:
        s = df[c]

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        mask &= (s.between(lower, upper)) | (s.isna())

    cleaned = df[mask]

    return cleaned, df.shape[0] - cleaned.shape[0]

# -------------------------------------------------------------
# Visualización básica
# -------------------------------------------------------------
def plot_histograms(df, cols, out_dir="plots"):
    """
    Genera histogramas simples para las variables numéricas.
    """
    os.makedirs(out_dir, exist_ok=True)

    for c in cols:
        plt.figure(figsize=(6, 4))
        plt.hist(df[c].dropna(), bins=30)
        plt.title(f"Histograma - {c}")
        plt.xlabel(c)
        plt.ylabel("Frecuencia")

        plt.savefig(os.path.join(out_dir, f"hist_{c}.png"),
                    bbox_inches="tight")
        plt.close()

def plot_scatter(df, x, y, out_dir="plots"):
    """
    Genera diagramas de dispersión para analizar relaciones.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y}")

    plt.savefig(os.path.join(out_dir, f"scatter_{x}_vs_{y}.png"),
                bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# Codificación de variables categóricas
# -------------------------------------------------------------
def label_encode_columns(df, cols):
    """
    Aplica Label Encoding y guarda el mapeo real
    categoría -> número.
    """
    df = df.copy()
    maps = {}

    for c in cols:
        le = LabelEncoder()
        values = df[c].astype(str).fillna("NA")
        df[c + "_label"] = le.fit_transform(values)

        maps[c] = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, maps

def one_hot_encode(df, cols):
    """
    Aplica One Hot Encoding a columnas de baja cardinalidad.
    """
    return pd.get_dummies(df, columns=cols, dummy_na=True)

def binary_encode(df, col):
    """
    Codificación binaria simple basada en factorización.
    """
    df = df.copy()

    codes, _ = pd.factorize(df[col].astype(str).fillna("NA"))
    n_bits = int(np.ceil(np.log2(max(codes.max() + 1, 2))))

    for b in range(n_bits):
        df[f"{col}_bin_{b}"] = ((codes >> b) & 1)

    return df

# -------------------------------------------------------------
# Reducción por alta correlación
# -------------------------------------------------------------
def drop_highly_correlated(df, threshold=0.9):
    """
    Elimina variables altamente correlacionadas.
    """
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [
        c for c in upper.columns
        if any(upper[c] > threshold)
    ]

    return df.drop(columns=to_drop), to_drop

# -------------------------------------------------------------
# Escalado
# -------------------------------------------------------------
def scale_columns(df, cols):
    """
    Aplica MinMaxScaler y StandardScaler por columna,
    sin eliminar filas completas por valores faltantes.
    """
    df = df.copy()

    for c in cols:
        values = df[[c]]

        if values.dropna().shape[0] == 0:
            # columna completamente vacía
            continue

        mm = MinMaxScaler()
        ss = StandardScaler()

        mask = values[c].notna()

        df.loc[mask, c + "_minmax"] = mm.fit_transform(values.loc[mask])
        df.loc[mask, c + "_std"] = ss.fit_transform(values.loc[mask])

    return df

# -------------------------------------------------------------
# Transformación logarítmica según sesgo
# -------------------------------------------------------------
def log_transform_if_skewed(df, cols, skew_threshold=1.0):
    """
    Aplica log(1+x) si la distribución es muy asimétrica.
    """
    df = df.copy()

    for c in cols:
        s = df[c].dropna()

        if not s.empty and abs(s.skew()) > skew_threshold:
            df[c + "_log"] = np.log1p(df[c].clip(lower=0))

    return df

# -------------------------------------------------------------
# Flujo principal
# -------------------------------------------------------------
def main():

    base = os.path.dirname(__file__)
    path = os.path.join(base, "movies.csv")

    print("Leyendo archivo:", path)

    df = read_movies(path)
    df = clean_columns(df)

    print("\nColumnas disponibles:")
    print(df.columns.tolist())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("\nColumnas numéricas:", numeric_cols)

    # Estadística descriptiva
    print("\n--- Tendencia central ---")
    print(central_tendency(df, numeric_cols))

    print("\n--- Dispersión ---")
    print(dispersion_measures(df, numeric_cols))

    print("\n--- Posición ---")
    print(position_measures(df, numeric_cols))

    # Outliers
    cols_out = [c for c in ["RunTime", "VOTES", "Gross", "RATING"]
                if c in df.columns]

    cleaned_df, removed = remove_outliers_iqr(df, cols_out)

    print(f"\nFilas eliminadas por outliers: {removed}")

    # Gráficos
    plot_histograms(df, numeric_cols)

    if "RunTime" in df.columns and "RATING" in df.columns:
        plot_scatter(df, "RunTime", "RATING")

    if "VOTES" in df.columns and "RATING" in df.columns:
        plot_scatter(df, "VOTES", "RATING")

    # Codificación
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if cat_cols:
        df_label, maps = label_encode_columns(df, cat_cols)
        print("\nEjemplo de mapas de Label Encoding (solo 5 categorías):")

        for col in list(maps.keys())[:2]:
            sample = list(maps[col].items())[:5]
            print(f"Columna: {col} -> ejemplo:", sample)

        small = [c for c in cat_cols if df[c].nunique(dropna=False) <= 20]
        df_ohe = one_hot_encode(df, small)

        df_bin = binary_encode(df, cat_cols[0])

    # Correlación
    reduced, dropped = drop_highly_correlated(
        df[numeric_cols]
    )

    print("\nColumnas eliminadas por alta correlación:", dropped)

    # Escalado
    df_scaled = scale_columns(df, numeric_cols)

    # Transformaciones logarítmicas
    df_final = log_transform_if_skewed(df_scaled, numeric_cols)

    # Guardado
    cleaned_df.to_csv(os.path.join(base, "movies_cleaned.csv"), index=False)
    df_final.to_csv(os.path.join(base, "movies_transformed.csv"), index=False)

    print("\nArchivos generados correctamente.")

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------
# CONCLUSIONES
# -----------------------------------------------------------------
#
# 1. El conjunto de datos presenta problemas típicos de calidad:
#    valores faltantes, variables numéricas almacenadas como texto
#    y símbolos que impiden el análisis directo.
#
# 2. La estadística descriptiva permite entender la distribución
#    de variables relevantes como calificación, duración, votos
#    e ingresos.
#
# 3. Se aplicó el criterio IQR para la detección de outliers, con el
#    fin de reducir la influencia de valores extremos en modelos
#    sensibles a la escala.
#
# 4. Se exploraron distintas estrategias de codificación:
#    Label Encoding, One Hot Encoding y una codificación binaria
#    simple, dependiendo de la cardinalidad y del tipo de modelo.
#
# 5. Se aplicaron técnicas de escalado (Min-Max y estandarización),
#    necesarias para algoritmos como regresión, SVM y redes
#    neuronales.
#
# 6. Finalmente, se aplicaron transformaciones logarítmicas en
#    variables altamente sesgadas para estabilizar la varianza
#    y facilitar el aprendizaje del modelo.
#
# En general, este flujo representa un pipeline básico de
# preprocesamiento para un problema de Machine Learning
# supervisado con datos tabulares.