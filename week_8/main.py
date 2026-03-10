import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# =========================
# Cargar dataset
# =========================
df = pd.read_csv("Lecture08/dataset_sintetico_FIRE_UdeA_realista.csv")

print("Primeras filas")
print(df.head())

print("\nShape del dataset:", df.shape)

# =========================
# Información general
# =========================
print("\nInfo del dataset")
print(df.info())

print("\nEstadísticas descriptivas")
print(df.describe())

# =========================
# Valores faltantes
# =========================
missing = df.isnull().sum().sort_values(ascending=False)

print("\nValores faltantes")
print(missing)

plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Mapa de valores faltantes")
plt.show()

# =========================
# Distribución del label
# =========================
plt.figure(figsize=(6,4))
sns.countplot(x="label", data=df)
plt.title("Distribución del Riesgo Financiero")
plt.show()

print("\nDistribución porcentual")
print(df["label"].value_counts(normalize=True))

# =========================
# Variables numéricas
# =========================
numeric_cols = df.select_dtypes(include=np.number).columns

df[numeric_cols].hist(figsize=(14,10), bins=30)
plt.suptitle("Distribución de variables financieras")
plt.show()

# =========================
# Riesgo por unidad
# =========================
df["unidad"] = df["unidad"].astype(str)

plt.figure(figsize=(10,6))
sns.countplot(y="unidad", hue="label", data=df)
plt.title("Riesgo financiero por unidad")
plt.show()

# =========================
# Evolución temporal
# =========================
riesgo_anual = df.groupby("anio")["label"].mean()

plt.figure(figsize=(8,5))
riesgo_anual.plot(marker="o")
plt.title("Probabilidad de riesgo financiero por año")
plt.ylabel("Tasa de riesgo")
plt.show()

# =========================
# Liquidez vs riesgo
# =========================
plt.figure(figsize=(6,5))
sns.boxplot(x="label", y="liquidez", data=df)
plt.title("Liquidez vs Riesgo Financiero")
plt.show()

# =========================
# Días de efectivo vs riesgo
# =========================
plt.figure(figsize=(6,5))
sns.boxplot(x="label", y="dias_efectivo", data=df)
plt.title("Días de efectivo vs Riesgo")
plt.show()

# =========================
# CFO vs riesgo
# =========================
plt.figure(figsize=(6,5))
sns.boxplot(x="label", y="cfo", data=df)
plt.title("Flujo de caja operativo vs Riesgo")
plt.show()

# =========================
# Correlaciones
# =========================
corr = df[numeric_cols].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()

# Variables más correlacionadas con el label
corr_target = corr["label"].sort_values(ascending=False)

print("\nCorrelación con label")
print(corr_target)

# =========================
# HHI vs riesgo
# =========================
plt.figure(figsize=(6,5))
sns.boxplot(x="label", y="hhi_fuentes", data=df)
plt.title("Concentración de ingresos vs Riesgo")
plt.show()

# =========================
# Scatter liquidez vs efectivo
# =========================
plt.figure(figsize=(6,5))

sns.scatterplot(
    x="liquidez",
    y="dias_efectivo",
    hue="label",
    data=df
)

plt.title("Liquidez vs Días de efectivo")
plt.show()

# =========================
# Promedios por clase
# =========================
print("\nPromedios por label")
print(df.groupby("label")[numeric_cols].mean())

# =========================
# Outliers
# =========================
for col in numeric_cols:

    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# =========================
# Resumen dataset
# =========================
print("\nNúmero de unidades:", df["unidad"].nunique())
print("Número de años:", df["anio"].nunique())
print("Total observaciones:", len(df))