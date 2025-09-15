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
