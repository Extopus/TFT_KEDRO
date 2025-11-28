"""
Pipeline de Data Science para el proyecto TFT Kedro.

Este pipeline implementa modelos de Machine Learning para:
1. Clasificación de rangos competitivos (Challenger/Grandmaster/Platinum)
2. Regresión de placement en partidas (1-8)
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_ml_data,
    train_classification_model,
    train_regression_model,
    evaluate_feature_importance,
    generate_ml_insights,
    save_ml_models
)
from .integration_nodes import (
    add_clustering_features,
    compare_classification_with_clusters,
    compare_regression_with_clusters
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de Data Science para TFT.
    
    Returns:
        Pipeline de Kedro con nodos de Machine Learning
    """
    return Pipeline(
        [
            # Preparar datos para ML - Clasificación
            node(
                func=prepare_ml_data,
                inputs=["tft_combined_features", "params:ml_config"],
                outputs=["X_classification", "y_classification", "feature_names_class"],
                name="prepare_classification_data",
                tags=["ml", "classification", "data_preparation"]
            ),
            
            # Preparar datos para ML - Regresión
            node(
                func=prepare_ml_data,
                inputs=["tft_combined_features", "params:ml_config"],
                outputs=["X_regression", "y_regression", "feature_names_reg"],
                name="prepare_regression_data",
                tags=["ml", "regression", "data_preparation"]
            ),
            
            # Entrenar modelo de clasificación
            node(
                func=train_classification_model,
                inputs=["X_classification", "y_classification", "params:ml_config"],
                outputs="classification_results",
                name="train_classification_model",
                tags=["ml", "classification", "training"]
            ),
            
            # Entrenar modelo de regresión
            node(
                func=train_regression_model,
                inputs=["X_regression", "y_regression", "params:ml_config"],
                outputs="regression_results",
                name="train_regression_model",
                tags=["ml", "regression", "training"]
            ),
            
            # Evaluar importancia de features - Clasificación
            node(
                func=evaluate_feature_importance,
                inputs="classification_results",
                outputs="classification_feature_importance",
                name="evaluate_classification_features",
                tags=["ml", "classification", "feature_analysis"]
            ),
            
            # Evaluar importancia de features - Regresión
            node(
                func=evaluate_feature_importance,
                inputs="regression_results",
                outputs="regression_feature_importance",
                name="evaluate_regression_features",
                tags=["ml", "regression", "feature_analysis"]
            ),
            
            # Generar insights de ML
            node(
                func=generate_ml_insights,
                inputs=["classification_results", "regression_results"],
                outputs="ml_insights",
                name="generate_ml_insights",
                tags=["ml", "insights", "analysis"]
            ),
            
            # Guardar modelos entrenados
            node(
                func=save_ml_models,
                inputs=["classification_results", "regression_results", "params:ml_config"],
                outputs="saved_models_info",
                name="save_ml_models",
                tags=["ml", "model_persistence", "output"]
            ),
        ],
        tags="data_science"
    )


def create_integrated_pipeline(**kwargs) -> Pipeline:
    """
    Crea un pipeline que integra clustering con modelos supervisados.
    
    Returns:
        Pipeline de Kedro que integra unsupervised y supervised learning
    """
    return Pipeline(
        [
            # Agregar features de clustering al dataset
            node(
                func=add_clustering_features,
                inputs={
                    "df": "tft_combined_features",
                    "kmeans_results": "kmeans_results",
                    "dbscan_results": "dbscan_results",
                    "hierarchical_results": "hierarchical_results"
                },
                outputs="tft_features_with_clusters",
                name="add_clustering_features",
                tags=["ml", "integration", "feature_engineering"]
            ),
            
            # Comparar modelos con y sin clusters - Clasificación
            node(
                func=compare_classification_with_clusters,
                inputs={
                    "df_original": "tft_combined_features",
                    "df_with_clusters": "tft_features_with_clusters",
                    "params": "params:ml_config"
                },
                outputs="classification_integration_comparison",
                name="compare_classification_with_clusters",
                tags=["ml", "classification", "integration", "comparison"]
            ),
            
            # Comparar modelos con y sin clusters - Regresión
            node(
                func=compare_regression_with_clusters,
                inputs={
                    "df_original": "tft_combined_features",
                    "df_with_clusters": "tft_features_with_clusters",
                    "params": "params:ml_config"
                },
                outputs="regression_integration_comparison",
                name="compare_regression_with_clusters",
                tags=["ml", "regression", "integration", "comparison"]
            ),
        ],
        tags="data_science_integration"
    )
