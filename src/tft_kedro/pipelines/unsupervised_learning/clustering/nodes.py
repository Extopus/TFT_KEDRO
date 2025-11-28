"""
Nodos para el pipeline de Clustering.

Implementa múltiples algoritmos de clustering:
- K-Means
- DBSCAN
- Hierarchical Clustering
- Gaussian Mixture Models (opcional)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_data_for_clustering(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Prepara los datos para clustering.
    
    Args:
        df: DataFrame con features
        params: Parámetros de configuración
        
    Returns:
        Tuple con (datos escalados, scaler, nombres de features)
    """
    logger.info("Preparando datos para clustering...")
    
    # Obtener columnas numéricas (excluir IDs y targets)
    exclude_cols = params.get('exclude_columns', ['gameId', 'rank', 'Ranked'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Seleccionar y limpiar datos
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    
    logger.info(f"Datos preparados: {X_scaled_df.shape}")
    logger.info(f"Features utilizadas: {len(feature_cols)}")
    
    return X_scaled_df, scaler, feature_cols


def find_optimal_k_elbow(
    X: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Encuentra el K óptimo usando el método Elbow.
    
    Args:
        X: Datos escalados
        params: Parámetros de configuración
        
    Returns:
        Diccionario con K óptimo y métricas
    """
    logger.info("Calculando K óptimo con método Elbow...")
    
    k_range = params.get('k_range', range(2, 11))
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    # Encontrar K óptimo (máximo silhouette score)
    optimal_k_idx = np.argmax(silhouette_scores)
    optimal_k = list(k_range)[optimal_k_idx]
    
    results = {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k,
        'optimal_silhouette': silhouette_scores[optimal_k_idx]
    }
    
    logger.info(f"K óptimo encontrado: {optimal_k} (silhouette: {results['optimal_silhouette']:.4f})")
    
    return results


def apply_kmeans(
    X: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aplica K-Means clustering.
    
    Args:
        X: Datos escalados
        params: Parámetros de configuración
        
    Returns:
        Diccionario con modelo, labels y métricas
    """
    logger.info("Aplicando K-Means clustering...")
    
    n_clusters = params.get('kmeans_n_clusters', 3)
    random_state = params.get('random_state', 42)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    labels = kmeans.fit_predict(X)
    
    # Calcular métricas
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    results = {
        'algorithm': 'K-Means',
        'model': kmeans,
        'labels': labels,
        'n_clusters': n_clusters,
        'inertia': kmeans.inertia_,
        'silhouette_score': float(silhouette),
        'davies_bouldin_score': float(davies_bouldin),
        'calinski_harabasz_score': float(calinski_harabasz),
        'centroids': kmeans.cluster_centers_.tolist()
    }
    
    logger.info(f"K-Means completado: {n_clusters} clusters, silhouette={silhouette:.4f}")
    
    return results


def apply_dbscan(
    X: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aplica DBSCAN clustering.
    
    Args:
        X: Datos escalados
        params: Parámetros de configuración
        
    Returns:
        Diccionario con modelo, labels y métricas
    """
    logger.info("Aplicando DBSCAN clustering...")
    
    eps = params.get('dbscan_eps', 0.5)
    min_samples = params.get('dbscan_min_samples', 5)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Contar clusters (excluyendo ruido -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    results = {
        'algorithm': 'DBSCAN',
        'model': dbscan,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'eps': eps,
        'min_samples': min_samples
    }
    
    # Calcular métricas solo si hay al menos 2 clusters
    if n_clusters >= 2:
        # Filtrar ruido para métricas
        mask = labels != -1
        if mask.sum() > 0:
            X_filtered = X[mask]
            labels_filtered = labels[mask]
            
            results['silhouette_score'] = float(silhouette_score(X_filtered, labels_filtered))
            results['davies_bouldin_score'] = float(davies_bouldin_score(X_filtered, labels_filtered))
            results['calinski_harabasz_score'] = float(calinski_harabasz_score(X_filtered, labels_filtered))
        else:
            results['silhouette_score'] = None
            results['davies_bouldin_score'] = None
            results['calinski_harabasz_score'] = None
    else:
        results['silhouette_score'] = None
        results['davies_bouldin_score'] = None
        results['calinski_harabasz_score'] = None
    
    logger.info(f"DBSCAN completado: {n_clusters} clusters, {n_noise} puntos de ruido")
    
    return results


def apply_hierarchical(
    X: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aplica Hierarchical Clustering (Agglomerative).
    
    Para datasets grandes (>10k muestras), usa una submuestra para evitar problemas de memoria.
    
    Args:
        X: Datos escalados
        params: Parámetros de configuración
        
    Returns:
        Diccionario con modelo, labels y métricas
    """
    logger.info("Aplicando Hierarchical Clustering...")
    
    n_clusters = params.get('hierarchical_n_clusters', 3)
    linkage = params.get('hierarchical_linkage', 'ward')
    max_samples = params.get('hierarchical_max_samples', 10000)
    
    # Para datasets grandes, usar submuestra (Hierarchical es O(n²) en memoria)
    use_sample = len(X) > max_samples
    if use_sample:
        logger.warning(
            f"Dataset grande ({len(X)} muestras). "
            f"Usando submuestra de {max_samples} para Hierarchical Clustering "
            f"(requiere O(n²) memoria)."
        )
        # Usar RandomState para reproducibilidad
        rng = np.random.RandomState(params.get('random_state', 42))
        sample_indices = rng.choice(
            len(X), 
            size=max_samples, 
            replace=False
        )
        X_sample = X.iloc[sample_indices] if isinstance(X, pd.DataFrame) else X[sample_indices]
        original_indices = sample_indices
    else:
        X_sample = X
        original_indices = None
    
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    
    labels_sample = hierarchical.fit_predict(X_sample)
    
    # Si usamos submuestra, asignar labels a todas las muestras usando K-Means
    if use_sample:
        logger.info("Asignando clusters a todas las muestras usando centroides...")
        from sklearn.cluster import KMeans
        # Usar los centroides de los clusters encontrados
        centroids = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels_sample == cluster_id
            if cluster_mask.sum() > 0:
                centroid = X_sample[cluster_mask].mean(axis=0)
                centroids.append(centroid)
        
        if len(centroids) == n_clusters:
            # Asignar todas las muestras al cluster más cercano
            kmeans_assigner = KMeans(
                n_clusters=n_clusters,
                init=np.array(centroids),
                n_init=1,
                max_iter=1,
                random_state=params.get('random_state', 42)
            )
            labels = kmeans_assigner.fit_predict(X)
        else:
            # Fallback: usar K-Means normal
            logger.warning("No se pudieron calcular todos los centroides. Usando K-Means como fallback.")
            kmeans_assigner = KMeans(
                n_clusters=n_clusters,
                random_state=params.get('random_state', 42),
                n_init=10
            )
            labels = kmeans_assigner.fit_predict(X)
    else:
        labels = labels_sample
    
    # Calcular métricas
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    results = {
        'algorithm': 'Hierarchical',
        'model': hierarchical,
        'labels': labels,
        'n_clusters': n_clusters,
        'linkage': linkage,
        'silhouette_score': float(silhouette),
        'davies_bouldin_score': float(davies_bouldin),
        'calinski_harabasz_score': float(calinski_harabasz),
        'used_sample': use_sample,
        'sample_size': max_samples if use_sample else len(X)
    }
    
    logger.info(f"Hierarchical completado: {n_clusters} clusters, silhouette={silhouette:.4f}")
    if use_sample:
        logger.info(f"  (Usando submuestra de {max_samples} de {len(X)} muestras totales)")
    
    return results


def compare_clustering_algorithms(
    kmeans_results: Dict[str, Any],
    dbscan_results: Dict[str, Any],
    hierarchical_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compara los resultados de diferentes algoritmos de clustering.
    
    Args:
        kmeans_results: Resultados de K-Means
        dbscan_results: Resultados de DBSCAN
        hierarchical_results: Resultados de Hierarchical
        
    Returns:
        DataFrame con comparación de métricas
    """
    logger.info("Comparando algoritmos de clustering...")
    
    comparison_data = []
    
    # K-Means
    comparison_data.append({
        'Algorithm': 'K-Means',
        'N_Clusters': kmeans_results['n_clusters'],
        'Silhouette': kmeans_results['silhouette_score'],
        'Davies-Bouldin': kmeans_results['davies_bouldin_score'],
        'Calinski-Harabasz': kmeans_results['calinski_harabasz_score']
    })
    
    # DBSCAN
    if dbscan_results.get('silhouette_score') is not None:
        comparison_data.append({
            'Algorithm': 'DBSCAN',
            'N_Clusters': dbscan_results['n_clusters'],
            'Silhouette': dbscan_results['silhouette_score'],
            'Davies-Bouldin': dbscan_results['davies_bouldin_score'],
            'Calinski-Harabasz': dbscan_results['calinski_harabasz_score']
        })
    else:
        comparison_data.append({
            'Algorithm': 'DBSCAN',
            'N_Clusters': dbscan_results['n_clusters'],
            'Silhouette': None,
            'Davies-Bouldin': None,
            'Calinski-Harabasz': None
        })
    
    # Hierarchical
    comparison_data.append({
        'Algorithm': 'Hierarchical',
        'N_Clusters': hierarchical_results['n_clusters'],
        'Silhouette': hierarchical_results['silhouette_score'],
        'Davies-Bouldin': hierarchical_results['davies_bouldin_score'],
        'Calinski-Harabasz': hierarchical_results['calinski_harabasz_score']
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    logger.info("Comparación completada")
    logger.info(f"\n{comparison_df.to_string()}")
    
    return comparison_df


def analyze_cluster_patterns(
    df: pd.DataFrame,
    clustering_results: Dict[str, Any],
    feature_names: List[str],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analiza patrones y estadísticas por cluster.
    
    Args:
        df: DataFrame original con todas las columnas
        clustering_results: Diccionario con resultados del clustering (debe tener 'labels')
        feature_names: Nombres de las features usadas
        params: Parámetros de configuración
        
    Returns:
        Diccionario con análisis de patrones
    """
    # Extraer labels y nombre del algoritmo
    labels = clustering_results['labels']
    algorithm_name = clustering_results.get('algorithm', 'Unknown')
    
    logger.info(f"Analizando patrones de clusters para {algorithm_name}...")
    
    # Agregar labels al dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    # Estadísticas por cluster
    cluster_stats = {}
    cluster_profiles = {}
    
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:  # Ruido en DBSCAN
            cluster_name = 'Noise'
        else:
            cluster_name = f'Cluster_{cluster_id}'
        
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        # Estadísticas descriptivas
        stats = {
            'n_samples': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100
        }
        
        # Estadísticas por feature
        feature_stats = {}
        for feature in feature_names:
            if feature in cluster_data.columns:
                feature_stats[feature] = {
                    'mean': float(cluster_data[feature].mean()),
                    'std': float(cluster_data[feature].std()),
                    'median': float(cluster_data[feature].median()),
                    'min': float(cluster_data[feature].min()),
                    'max': float(cluster_data[feature].max())
                }
        
        stats['features'] = feature_stats
        
        # Si existe columna 'rank', analizar distribución
        if 'rank' in df_with_clusters.columns:
            rank_dist = cluster_data['rank'].value_counts(normalize=True).to_dict()
            stats['rank_distribution'] = {k: float(v) for k, v in rank_dist.items()}
        
        cluster_stats[cluster_name] = stats
        
        # Perfil del cluster (características más distintivas)
        if len(feature_names) > 0:
            cluster_mean = cluster_data[feature_names].mean()
            overall_mean = df[feature_names].mean()
            
            # Features que están por encima del promedio
            above_avg = (cluster_mean > overall_mean * 1.1).index.tolist()
            # Features que están por debajo del promedio
            below_avg = (cluster_mean < overall_mean * 0.9).index.tolist()
            
            cluster_profiles[cluster_name] = {
                'above_average_features': above_avg,
                'below_average_features': below_avg,
                'distinctive_features': above_avg + below_avg
            }
    
    results = {
        'algorithm': algorithm_name,
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'cluster_stats': cluster_stats,
        'cluster_profiles': cluster_profiles,
        'total_samples': len(df)
    }
    
    logger.info(f"Análisis completado: {results['n_clusters']} clusters analizados")
    
    return results

