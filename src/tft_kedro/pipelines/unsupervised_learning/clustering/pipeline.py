"""
Pipeline de Clustering para análisis no supervisado.

Este pipeline implementa múltiples algoritmos de clustering:
- K-Means
- DBSCAN
- Hierarchical Clustering
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_data_for_clustering,
    find_optimal_k_elbow,
    apply_kmeans,
    apply_dbscan,
    apply_hierarchical,
    compare_clustering_algorithms,
    analyze_cluster_patterns
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de clustering.
    
    Returns:
        Pipeline de Kedro con nodos de clustering
    """
    return Pipeline(
        [
            # Preparar datos para clustering
            node(
                func=prepare_data_for_clustering,
                inputs=["tft_combined_features", "params:unsupervised_config"],
                outputs=["clustering_data_scaled", "clustering_scaler", "clustering_feature_names"],
                name="prepare_clustering_data",
                tags=["unsupervised", "clustering", "data_preparation"]
            ),
            
            # Encontrar K óptimo (Elbow Method)
            node(
                func=find_optimal_k_elbow,
                inputs=["clustering_data_scaled", "params:unsupervised_config"],
                outputs="optimal_k_results",
                name="find_optimal_k",
                tags=["unsupervised", "clustering", "optimization"]
            ),
            
            # Aplicar K-Means
            node(
                func=apply_kmeans,
                inputs=["clustering_data_scaled", "params:unsupervised_config"],
                outputs="kmeans_results",
                name="apply_kmeans",
                tags=["unsupervised", "clustering", "kmeans"]
            ),
            
            # Aplicar DBSCAN
            node(
                func=apply_dbscan,
                inputs=["clustering_data_scaled", "params:unsupervised_config"],
                outputs="dbscan_results",
                name="apply_dbscan",
                tags=["unsupervised", "clustering", "dbscan"]
            ),
            
            # Aplicar Hierarchical Clustering
            node(
                func=apply_hierarchical,
                inputs=["clustering_data_scaled", "params:unsupervised_config"],
                outputs="hierarchical_results",
                name="apply_hierarchical",
                tags=["unsupervised", "clustering", "hierarchical"]
            ),
            
            # Comparar algoritmos
            node(
                func=compare_clustering_algorithms,
                inputs=["kmeans_results", "dbscan_results", "hierarchical_results"],
                outputs="clustering_comparison",
                name="compare_clustering_algorithms",
                tags=["unsupervised", "clustering", "comparison"]
            ),
            
            # Analizar patrones - K-Means
            node(
                func=analyze_cluster_patterns,
                inputs={
                    "df": "tft_combined_features",
                    "clustering_results": "kmeans_results",
                    "feature_names": "clustering_feature_names",
                    "params": "params:unsupervised_config"
                },
                outputs="kmeans_patterns",
                name="analyze_kmeans_patterns",
                tags=["unsupervised", "clustering", "pattern_analysis"]
            ),
            
            # Analizar patrones - DBSCAN
            node(
                func=analyze_cluster_patterns,
                inputs={
                    "df": "tft_combined_features",
                    "clustering_results": "dbscan_results",
                    "feature_names": "clustering_feature_names",
                    "params": "params:unsupervised_config"
                },
                outputs="dbscan_patterns",
                name="analyze_dbscan_patterns",
                tags=["unsupervised", "clustering", "pattern_analysis"]
            ),
            
            # Analizar patrones - Hierarchical
            node(
                func=analyze_cluster_patterns,
                inputs={
                    "df": "tft_combined_features",
                    "clustering_results": "hierarchical_results",
                    "feature_names": "clustering_feature_names",
                    "params": "params:unsupervised_config"
                },
                outputs="hierarchical_patterns",
                name="analyze_hierarchical_patterns",
                tags=["unsupervised", "clustering", "pattern_analysis"]
            ),
        ],
        tags="clustering"
    )

