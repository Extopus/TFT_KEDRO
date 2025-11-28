"""
Nodos para integrar resultados de aprendizaje no supervisado con modelos supervisados.

Este módulo contiene funciones para usar clusters como features adicionales
en modelos de machine learning supervisado.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
import joblib

logger = logging.getLogger(__name__)


def add_clustering_features(
    df: pd.DataFrame,
    kmeans_results: Dict[str, Any],
    dbscan_results: Dict[str, Any],
    hierarchical_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Agrega labels de clustering como features adicionales al dataset.
    
    Args:
        df: DataFrame original con features
        kmeans_results: Resultados de K-Means clustering
        dbscan_results: Resultados de DBSCAN clustering
        hierarchical_results: Resultados de Hierarchical clustering
        
    Returns:
        DataFrame con features de clustering agregadas
    """
    logger.info("Agregando features de clustering al dataset...")
    
    df_with_clusters = df.copy()
    
    # Extraer labels de cada algoritmo
    kmeans_labels = kmeans_results.get('labels', np.array([]))
    dbscan_labels = dbscan_results.get('labels', np.array([]))
    hierarchical_labels = hierarchical_results.get('labels', np.array([]))
    
    # Agregar labels como features categóricas (one-hot encoding)
    if len(kmeans_labels) > 0:
        df_with_clusters['cluster_kmeans'] = kmeans_labels
        logger.info(f"  ✓ Agregado cluster_kmeans: {len(np.unique(kmeans_labels))} clusters")
    
    if len(dbscan_labels) > 0:
        df_with_clusters['cluster_dbscan'] = dbscan_labels
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        logger.info(f"  ✓ Agregado cluster_dbscan: {n_clusters_dbscan} clusters")
    
    if len(hierarchical_labels) > 0:
        df_with_clusters['cluster_hierarchical'] = hierarchical_labels
        logger.info(f"  ✓ Agregado cluster_hierarchical: {len(np.unique(hierarchical_labels))} clusters")
    
    # Crear features one-hot encoded para clusters (opcional, pero útil para algunos modelos)
    # Por ahora solo agregamos los labels como features numéricas
    
    logger.info(f"Dataset actualizado: {df_with_clusters.shape} (original: {df.shape})")
    
    return df_with_clusters


def compare_classification_with_clusters(
    df_original: pd.DataFrame,
    df_with_clusters: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compara el rendimiento de modelos de clasificación con y sin features de clustering.
    
    Args:
        df_original: DataFrame original sin features de clustering
        df_with_clusters: DataFrame con features de clustering agregadas
        params: Parámetros de configuración
        
    Returns:
        Diccionario con comparación de métricas
    """
    target_type = 'classification'
    logger.info(f"Comparando modelos de clasificación con y sin clusters...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    
    # Preparar datos originales
    exclude_cols = ['gameId', 'rank', 'Ranked', 'cluster_kmeans', 'cluster_dbscan', 'cluster_hierarchical']
    numeric_cols_orig = df_original.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols_orig = [col for col in numeric_cols_orig if col not in exclude_cols]
    
    X_orig = df_original[feature_cols_orig].fillna(df_original[feature_cols_orig].median())
    
    # Determinar target según tipo
    if target_type == "classification":
        if 'rank' not in df_original.columns:
            raise ValueError("Columna 'rank' no encontrada para clasificación")
        y = df_original['rank']
        model_orig = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        metric_name = 'accuracy'
    else:
        if 'Ranked' not in df_original.columns:
            raise ValueError("Columna 'Ranked' no encontrada para regresión")
        y = df_original['Ranked']
        model_orig = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        metric_name = 'r2_score'
    
    # Entrenar modelo sin clusters
    stratify_param = y if target_type == "classification" else None
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_orig, y, test_size=params.get('test_size', 0.2), 
        random_state=params.get('random_state', 42), 
        stratify=stratify_param
    )
    model_orig.fit(X_train_orig, y_train)
    y_pred_orig = model_orig.predict(X_test_orig)
    
    if target_type == "classification":
        metric_orig = accuracy_score(y_test, y_pred_orig)
    else:
        metric_orig = r2_score(y_test, y_pred_orig)
    
    # Preparar datos con clusters
    numeric_cols_clust = df_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols_clust = [col for col in numeric_cols_clust if col not in ['gameId', 'rank', 'Ranked']]
    
    X_clust = df_with_clusters[feature_cols_clust].fillna(df_with_clusters[feature_cols_clust].median())
    
    # Entrenar modelo con clusters (usar mismos parámetros para comparación justa)
    X_train_clust, X_test_clust, _, _ = train_test_split(
        X_clust, y, test_size=params.get('test_size', 0.2), 
        random_state=params.get('random_state', 42), 
        stratify=stratify_param
    )
    
    if target_type == "classification":
        model_clust = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model_clust = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    model_clust.fit(X_train_clust, y_train)
    y_pred_clust = model_clust.predict(X_test_clust)
    
    if target_type == "classification":
        metric_clust = accuracy_score(y_test, y_pred_clust)
    else:
        metric_clust = r2_score(y_test, y_pred_clust)
    
    # Calcular mejora
    improvement = ((metric_clust - metric_orig) / metric_orig) * 100 if metric_orig > 0 else 0
    
    results = {
        'target_type': target_type,
        'metric_name': metric_name,
        'without_clusters': {
            'metric': float(metric_orig),
            'n_features': len(feature_cols_orig)
        },
        'with_clusters': {
            'metric': float(metric_clust),
            'n_features': len(feature_cols_clust)
        },
        'improvement': float(improvement),
        'improved': metric_clust > metric_orig
    }
    
    logger.info(f"  Sin clusters: {metric_name}={metric_orig:.4f} ({len(feature_cols_orig)} features)")
    logger.info(f"  Con clusters: {metric_name}={metric_clust:.4f} ({len(feature_cols_clust)} features)")
    logger.info(f"  Mejora: {improvement:+.2f}%")
    
    return results


def compare_regression_with_clusters(
    df_original: pd.DataFrame,
    df_with_clusters: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compara el rendimiento de modelos de regresión con y sin features de clustering.
    
    Args:
        df_original: DataFrame original sin features de clustering
        df_with_clusters: DataFrame con features de clustering agregadas
        params: Parámetros de configuración
        
    Returns:
        Diccionario con comparación de métricas
    """
    target_type = 'regression'
    logger.info(f"Comparando modelos de regresión con y sin clusters...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Preparar datos originales
    exclude_cols = ['gameId', 'rank', 'Ranked', 'cluster_kmeans', 'cluster_dbscan', 'cluster_hierarchical']
    numeric_cols_orig = df_original.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols_orig = [col for col in numeric_cols_orig if col not in exclude_cols]
    
    X_orig = df_original[feature_cols_orig].fillna(df_original[feature_cols_orig].median())
    
    if 'Ranked' not in df_original.columns:
        raise ValueError("Columna 'Ranked' no encontrada para regresión")
    y = df_original['Ranked']
    model_orig = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    metric_name = 'r2_score'
    
    # Entrenar modelo sin clusters
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_orig, y, test_size=params.get('test_size', 0.2), 
        random_state=params.get('random_state', 42)
    )
    model_orig.fit(X_train_orig, y_train)
    y_pred_orig = model_orig.predict(X_test_orig)
    metric_orig = r2_score(y_test, y_pred_orig)
    
    # Preparar datos con clusters
    numeric_cols_clust = df_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols_clust = [col for col in numeric_cols_clust if col not in ['gameId', 'rank', 'Ranked']]
    
    X_clust = df_with_clusters[feature_cols_clust].fillna(df_with_clusters[feature_cols_clust].median())
    
    # Entrenar modelo con clusters
    X_train_clust, X_test_clust, _, _ = train_test_split(
        X_clust, y, test_size=params.get('test_size', 0.2), 
        random_state=params.get('random_state', 42)
    )
    
    model_clust = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_clust.fit(X_train_clust, y_train)
    y_pred_clust = model_clust.predict(X_test_clust)
    metric_clust = r2_score(y_test, y_pred_clust)
    
    # Calcular mejora
    improvement = ((metric_clust - metric_orig) / abs(metric_orig)) * 100 if metric_orig != 0 else 0
    
    results = {
        'target_type': target_type,
        'metric_name': metric_name,
        'without_clusters': {
            'metric': float(metric_orig),
            'n_features': len(feature_cols_orig)
        },
        'with_clusters': {
            'metric': float(metric_clust),
            'n_features': len(feature_cols_clust)
        },
        'improvement': float(improvement),
        'improved': metric_clust > metric_orig
    }
    
    logger.info(f"  Sin clusters: {metric_name}={metric_orig:.4f} ({len(feature_cols_orig)} features)")
    logger.info(f"  Con clusters: {metric_name}={metric_clust:.4f} ({len(feature_cols_clust)} features)")
    logger.info(f"  Mejora: {improvement:+.2f}%")
    
    return results

