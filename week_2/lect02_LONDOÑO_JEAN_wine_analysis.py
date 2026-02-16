# Jean Carlo Londoño Ocampo 

# Dataset elegido: Wine Dataset (scikit-learn) 

# ¿Por qué este?

# - Clasificación **supervisada multiclase**
# - Un poco más realista que Iris
# - Más features → mejor para ver:
   # - escalado
  #  - correlaciones
 #   - overfitting
# Sigue siendo pequeño y manejable

from sklearn.datasets import load_wine
import pandas as pd

# Recolección y exploración de datos

# Cargar dataset
wine = load_wine()

# Convertir a DataFrame
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="class")

# Ver primeras filas
X.head()

# Análisis geométrico rápido

# Información general
X.info()

# Distribución de clases
y.value_counts()

print("Dimensión del espacio:", X.shape[1])
print("Número de muestras:", X.shape[0])

# Preparación de datos (scaling)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# División train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# Evaluación del modelo

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Validación cruzada (anti-overfitting)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X_train_scaled, y_train, cv=5, scoring="accuracy"
)

print("Accuracy por fold:", scores)
print("Accuracy promedio:", scores.mean())

# Extraer los coeficientes del modelo

import pandas as pd
import numpy as np

# Obtener coeficientes
coef = model.coef_

# Convertir a DataFrame
coef_df = pd.DataFrame(
    coef,
    columns=X.columns,
    index=[f"Clase {i}" for i in np.unique(y)]
)

coef_df

# Importancia global (valor absoluto)
importance = np.abs(coef_df).mean(axis=0)

importance_df = importance.sort_values(ascending=False)
importance_df

# Visualizar las más importantes
import matplotlib.pyplot as plt

importance_df.head(10).plot(kind="bar")
plt.title("Importancia de las features (Regresión Logística)")
plt.ylabel("Importancia promedio")
plt.show()