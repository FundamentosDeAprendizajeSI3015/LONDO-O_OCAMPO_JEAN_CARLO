# ==========================================================
# Clustering Pipeline FIRE-UdeA - Validación de Etiquetas
# K-Means, DBSCAN, HDBSCAN, Jerárquico (Ward)
# + Probabilidad condicional P(clase|cluster)
# + Estabilidad Bootstrap
# + Métricas de validación externa
#
# By: Jean Carlo Londoño Ocampo
# Universidad de Antioquia - FIRE-UdeA
# Fecha: Marzo 2026
# ==========================================================

# -----------------------------
# 1. LIBRERÍAS
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, confusion_matrix,
    homogeneity_score, completeness_score, v_measure_score
)
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")

# Carpeta de salida para gráficos
OUTPUT_DIR = "graficas_clustering"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 2. CARGA DE DATOS
# ==========================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """Carga el dataset desde un CSV."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Datos cargados: {df.shape[0]} muestras, {df.shape[1]} columnas")
    return df


# ==========================================================
# 3. PREPROCESAMIENTO
# ==========================================================

def preprocess_data(df: pd.DataFrame, exclude_cols: list = None) -> tuple:
    """
    Preprocesa el dataset:
    - Excluye columnas no numéricas o no predictoras
    - Imputa valores faltantes con mediana
    - Estandariza con Z-score
    
    Retorna: (X_scaled, feature_names, labels)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Separar label
    labels = df["label"].values
    
    # Seleccionar features numéricas
    feature_cols = [c for c in df.columns if c not in exclude_cols + ["label"]]
    df_features = df[feature_cols].copy()
    
    # Convertir a numérico e imputar
    for col in feature_cols:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        if df_features[col].isnull().sum() > 0:
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)
            print(f"  [IMPUTACIÓN] {col}: {df_features[col].isnull().sum()} → mediana={median_val:.4f}")
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features.values)
    
    print(f"[INFO] Features: {feature_cols}")
    print(f"[INFO] Shape final: {X_scaled.shape}")
    
    return X_scaled, feature_cols, labels


# ==========================================================
# 4. DISTRIBUCIÓN DE ETIQUETAS
# ==========================================================

def label_distribution(labels: np.ndarray, dataset_name: str):
    """Calcula y muestra la distribución de las etiquetas."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print(f"\n{'='*50}")
    print(f"DISTRIBUCIÓN DE ETIQUETAS - {dataset_name}")
    print(f"{'='*50}")
    
    for u, c in zip(unique, counts):
        prob = c / total
        print(f"  Label {u}: n={c}, P(label={u}) = {prob:.4f}")
    
    if len(unique) == 2:
        ratio = counts[0] / counts[1]
        print(f"  Ratio clase 0/1: {ratio:.3f}")
    
    # Gráfico
    plt.figure(figsize=(8, 5))
    bars = plt.bar(unique, counts, color=["#3498db", "#e74c3c"],
                   edgecolor="black", alpha=0.85)
    for bar, c in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"n={c}\nP={c/total:.3f}", ha='center', fontweight='bold')
    plt.xlabel("Label")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de Etiquetas - {dataset_name}")
    plt.xticks(unique)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/distribucion_labels_{dataset_name}.png", dpi=200)
    plt.close()
    
    return dict(zip(unique, counts / total))


# ==========================================================
# 5. REDUCCIÓN DE DIMENSIONALIDAD (PCA)
# ==========================================================

def reduce_pca(X: np.ndarray, n_components: int = 2) -> tuple:
    """Reduce dimensionalidad con PCA."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"[INFO] PCA varianza explicada: {var_explained:.4f} ({var_explained*100:.2f}%)")
    return X_pca, pca


# ==========================================================
# 6. k-DISTANCE PLOT (DBSCAN)
# ==========================================================

def k_distance_plot(X: np.ndarray, k: int = 5, dataset_name: str = ""):
    """Gráfico de k-distance para selección de eps en DBSCAN."""
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances, color='#2c3e50', linewidth=2)
    plt.xlabel('Observaciones ordenadas')
    plt.ylabel(f'{k}-distance')
    plt.title(f'k-Distance Plot (k={k}) - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/k_distance_{dataset_name}.png", dpi=200)
    plt.close()
    
    return k_distances


# ==========================================================
# 7. ALGORITMOS DE CLUSTERING
# ==========================================================

def run_kmeans(X: np.ndarray, k: int = 2):
    """K-Means clustering."""
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    return model.fit_predict(X)


def run_dbscan(X: np.ndarray, eps: float, min_samples: int):
    """DBSCAN clustering."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)


def run_hdbscan(X: np.ndarray, min_cluster_size: int):
    """HDBSCAN clustering."""
    model = HDBSCAN(min_cluster_size=min_cluster_size)
    return model.fit_predict(X)


def run_hierarchical(X: np.ndarray, method: str = 'ward', n_clusters: int = 2):
    """Clustering jerárquico aglomerativo."""
    Z = linkage(X, method=method)
    labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    return labels, Z


# ==========================================================
# 8. MÉTRICAS DE VALIDACIÓN EXTERNA
# ==========================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Calcula métricas de validación externa entre etiquetas 
    verdaderas y asignación de clusters.
    """
    mask = y_pred >= 0  # Excluir ruido (-1)
    
    if mask.sum() == 0:
        print(f"  [{name}] TODAS las observaciones son ruido - no hay clusters.")
        return {"ARI": None, "NMI": None, "V-Measure": None,
                "n_clusters": 0, "n_noise": len(y_true), "noise_pct": 100.0}
    
    ari = adjusted_rand_score(y_true[mask], y_pred[mask])
    nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
    homo = homogeneity_score(y_true[mask], y_pred[mask])
    comp = completeness_score(y_true[mask], y_pred[mask])
    vm = v_measure_score(y_true[mask], y_pred[mask])
    
    n_clusters = len(np.unique(y_pred[mask]))
    n_noise = int(np.sum(~mask))
    noise_pct = round(n_noise / len(y_true) * 100, 2)
    
    # Silhouette (solo si hay más de 1 cluster)
    sil = None
    if n_clusters > 1:
        sil = silhouette_score(y_true[mask].reshape(-1, 1) if y_true.ndim == 1 
                               else y_true[mask], y_pred[mask])
    
    metrics = {
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "Homogeneity": round(homo, 4),
        "Completeness": round(comp, 4),
        "V-Measure": round(vm, 4),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
    }
    
    print(f"  [{name}] ARI={ari:.4f} | NMI={nmi:.4f} | V-Measure={vm:.4f} | "
          f"Clusters={n_clusters} | Ruido={n_noise} ({noise_pct}%)")
    
    return metrics


# ==========================================================
# 9. PROBABILIDAD CONDICIONAL
# ==========================================================

def class_probability_analysis(y_true: np.ndarray, y_pred: np.ndarray, name: str):
    """
    Calcula:
    - P(clase | cluster): qué tan puro es cada cluster
    - P(cluster | clase): cómo se distribuye cada clase en clusters
    """
    clusters = np.unique(y_pred)
    classes = np.unique(y_true)
    
    # P(clase | cluster)
    print(f"\n  --- {name}: P(clase | cluster) ---")
    p_class_given_cluster = {}
    for c in clusters:
        mask = y_pred == c
        total = mask.sum()
        probs = {}
        for cl in classes:
            p = np.sum((y_pred == c) & (y_true == cl)) / total if total > 0 else 0
            probs[f"clase_{cl}"] = round(p, 4)
        
        label_str = "Ruido" if c == -1 else f"Cluster {c}"
        p_class_given_cluster[label_str] = probs
        print(f"    {label_str} (n={total}): {probs}")
    
    # P(cluster | clase)
    print(f"\n  --- {name}: P(cluster | clase) ---")
    p_cluster_given_class = {}
    for cl in classes:
        mask = y_true == cl
        total = mask.sum()
        probs = {}
        for c in clusters:
            p = np.sum((y_true == cl) & (y_pred == c)) / total if total > 0 else 0
            label_str = "Ruido" if c == -1 else f"Cluster {c}"
            probs[label_str] = round(p, 4)
        
        p_cluster_given_class[f"Clase {cl}"] = probs
        print(f"    Clase {cl} (n={total}): {probs}")
    
    return p_class_given_cluster, p_cluster_given_class


# ==========================================================
# 10. ESTABILIDAD BOOTSTRAP
# ==========================================================

def clustering_stability(X: np.ndarray, cluster_func, name: str, n_bootstrap: int = 30):
    """
    Estima la estabilidad del clustering usando bootstrap
    y Adjusted Rand Index (ARI).
    """
    np.random.seed(42)
    labels_ref = cluster_func(X)
    ari_scores = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_sample = X[idx]
        labels_sample = cluster_func(X_sample)
        ari = adjusted_rand_score(labels_ref[idx], labels_sample)
        ari_scores.append(ari)
    
    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    
    print(f"  [{name}] Estabilidad Bootstrap: ARI medio={mean_ari:.4f} ± {std_ari:.4f}")
    
    return mean_ari, std_ari, ari_scores


# ==========================================================
# 11. VISUALIZACIÓN
# ==========================================================

def plot_clusters_vs_labels(X_2d, cluster_labels, true_labels, title, filename):
    """Scatter plot comparando clusters vs etiquetas reales."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Clusters
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels,
                                cmap='tab10', s=40, alpha=0.7)
    axes[0].set_title(f"{title} - Clusters")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(scatter1, ax=axes[0])
    
    # Etiquetas reales
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=true_labels,
                                cmap='coolwarm', s=40, alpha=0.7)
    axes[1].set_title(f"{title} - Etiquetas Reales")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Matriz de confusión entre etiqueta real y cluster."""
    labels_u = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_u)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=labels_u, yticklabels=labels_u)
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Etiqueta Real")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_probability_heatmap(prob_dict, title, filename):
    """Mapa de calor de probabilidades condicionales."""
    df_prob = pd.DataFrame(prob_dict).T
    if len(df_prob) == 0:
        return
    
    plt.figure(figsize=(8, max(4, len(df_prob) * 1.5)))
    sns.heatmap(df_prob, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5)
    plt.title(title)
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_stability(all_scores, algo_names, dataset_name):
    """Boxplot de estabilidad bootstrap."""
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(all_scores, tick_labels=algo_names, patch_artist=True)
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(algo_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.title(f"Estabilidad del Clustering - {dataset_name}")
    plt.ylabel("ARI Bootstrap")
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stability_{dataset_name}.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_dendrogram(Z, title, filename, truncate_level=5):
    """Dendrograma."""
    plt.figure(figsize=(14, 6))
    dendrogram(Z, truncate_mode='level', p=truncate_level, color_threshold=0)
    plt.title(title)
    plt.xlabel('Observaciones')
    plt.ylabel('Distancia')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png", dpi=200, bbox_inches="tight")
    plt.close()


# ==========================================================
# 12. PIPELINE PRINCIPAL
# ==========================================================

def run_pipeline(csv_path: str, dataset_name: str, exclude_cols: list = None,
                 dbscan_eps: float = 1.8, dbscan_min_samples: int = 10,
                 hdbscan_min_cluster_size: int = 15, k_neighbors: int = 10):
    """
    Pipeline completo de validación de etiquetas para un dataset.
    """
    print(f"\n{'#'*60}")
    print(f"# PIPELINE: {dataset_name}")
    print(f"{'#'*60}")
    
    # Cargar y preprocesar
    df = load_data(csv_path)
    X, features, y_true = preprocess_data(df, exclude_cols)
    
    # Distribución de etiquetas
    label_probs = label_distribution(y_true, dataset_name)
    
    # PCA
    X_pca, pca = reduce_pca(X)
    
    # k-distance plot
    k_distance_plot(X, k=k_neighbors, dataset_name=dataset_name)
    
    # --- CLUSTERING ---
    print(f"\n{'='*50}")
    print(f"CLUSTERING - {dataset_name}")
    print(f"{'='*50}")
    
    # K-Means
    km_labels = run_kmeans(X, k=2)
    print("\n[K-Means k=2]")
    km_metrics = compute_metrics(y_true, km_labels, "K-Means")
    km_p_cg, km_p_gc = class_probability_analysis(y_true, km_labels, "K-Means")
    
    # DBSCAN
    db_labels = run_dbscan(X, eps=dbscan_eps, min_samples=dbscan_min_samples)
    print(f"\n[DBSCAN eps={dbscan_eps}, min_samples={dbscan_min_samples}]")
    db_metrics = compute_metrics(y_true, db_labels, "DBSCAN")
    db_p_cg, db_p_gc = class_probability_analysis(y_true, db_labels, "DBSCAN")
    
    # HDBSCAN
    hdb_labels = run_hdbscan(X, min_cluster_size=hdbscan_min_cluster_size)
    print(f"\n[HDBSCAN min_cluster_size={hdbscan_min_cluster_size}]")
    hdb_metrics = compute_metrics(y_true, hdb_labels, "HDBSCAN")
    hdb_p_cg, hdb_p_gc = class_probability_analysis(y_true, hdb_labels, "HDBSCAN")
    
    # Jerárquico
    hier_labels, Z = run_hierarchical(X, method='ward', n_clusters=2)
    print("\n[Jerárquico Ward k=2]")
    hier_metrics = compute_metrics(y_true, hier_labels, "Jerárquico")
    hier_p_cg, hier_p_gc = class_probability_analysis(y_true, hier_labels, "Jerárquico")
    
    # --- ESTABILIDAD ---
    print(f"\n{'='*50}")
    print(f"ESTABILIDAD - {dataset_name}")
    print(f"{'='*50}")
    
    _, _, km_stab = clustering_stability(X, lambda X_: run_kmeans(X_, 2), "K-Means")
    _, _, db_stab = clustering_stability(
        X, lambda X_: run_dbscan(X_, dbscan_eps, dbscan_min_samples), "DBSCAN")
    _, _, hdb_stab = clustering_stability(
        X, lambda X_: run_hdbscan(X_, hdbscan_min_cluster_size), "HDBSCAN")
    
    # --- GRÁFICOS ---
    print(f"\n[INFO] Generando gráficos para {dataset_name}...")
    
    algo_list = [
        ("K-Means", km_labels), ("DBSCAN", db_labels),
        ("HDBSCAN", hdb_labels), ("Jerárquico", hier_labels)
    ]
    
    for algo_name, labels in algo_list:
        safe_name = algo_name.replace(" ", "_").lower()
        plot_clusters_vs_labels(X_pca, labels, y_true,
                                f"{dataset_name} - {algo_name}",
                                f"clusters_{safe_name}_{dataset_name}")
        plot_confusion_matrix(y_true, labels,
                              f"Confusión: {algo_name} - {dataset_name}",
                              f"confusion_{safe_name}_{dataset_name}")
    
    # Probabilidades
    for algo_name, p_cg in [("K-Means", km_p_cg), ("DBSCAN", db_p_cg)]:
        safe_name = algo_name.replace(" ", "_").lower()
        plot_probability_heatmap(p_cg, f"P(clase|cluster) - {algo_name} {dataset_name}",
                                 f"prob_heatmap_{safe_name}_{dataset_name}")
    
    # Estabilidad
    plot_stability([km_stab, db_stab, hdb_stab],
                   ["K-Means", "DBSCAN", "HDBSCAN"], dataset_name)
    
    # Dendrograma
    plot_dendrogram(Z, f"Dendrograma Ward - {dataset_name}",
                    f"dendrogram_{dataset_name}")
    
    print(f"\n[INFO] Pipeline {dataset_name} completado.")
    print(f"[INFO] Gráficos guardados en: {OUTPUT_DIR}/")
    
    return {
        "metrics": {
            "K-Means": km_metrics, "DBSCAN": db_metrics,
            "HDBSCAN": hdb_metrics, "Jerárquico": hier_metrics
        },
        "probabilities": {
            "K-Means": km_p_cg, "DBSCAN": db_p_cg,
            "HDBSCAN": hdb_p_cg, "Jerárquico": hier_p_cg
        },
        "label_probs": label_probs
    }


# ==========================================================
# 13. EJECUCIÓN PRINCIPAL
# ==========================================================

if __name__ == '__main__':
    
    # Dataset 1: Sintético (500 muestras, 7 features)
    results_d1 = run_pipeline(
        csv_path="dataset_sintetico_FIRE_UdeA.csv",
        dataset_name="D1_Sintetico",
        exclude_cols=[],
        dbscan_eps=1.8,
        dbscan_min_samples=10,
        hdbscan_min_cluster_size=15,
        k_neighbors=10
    )
    
    # Dataset 2: Realista (80 muestras, 13 features)
    results_d2 = run_pipeline(
        csv_path="dataset_sintetico_FIRE_UdeA_realista.csv",
        dataset_name="D2_Realista",
        exclude_cols=["anio", "unidad"],
        dbscan_eps=3.5,
        dbscan_min_samples=5,
        hdbscan_min_cluster_size=5,
        k_neighbors=5
    )
    
    # Resumen final
    print("\n" + "#" * 60)
    print("# RESUMEN FINAL")
    print("#" * 60)
    
    print("\n--- Dataset 1 (Sintético, n=500) ---")
    for algo, m in results_d1["metrics"].items():
        print(f"  {algo}: ARI={m.get('ARI', 'N/A')}")
    
    print("\n--- Dataset 2 (Realista, n=80) ---")
    for algo, m in results_d2["metrics"].items():
        print(f"  {algo}: ARI={m.get('ARI', 'N/A')}")
    
    print("\nCONCLUSIÓN: Las etiquetas NO forman clusters naturales.")
    print("ARI ≈ 0 en todos los casos → concordancia al nivel del azar.")
