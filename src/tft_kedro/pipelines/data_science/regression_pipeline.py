"""
Pipeline de Regresión para TFT Kedro.

Este pipeline implementa modelos de regresión para predecir variables numéricas
como puntaje de LP o posición final.
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_ml_data,
    train_regression_model,
    evaluate_feature_importance,
    save_ml_models
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de regresión.
    
    Returns:
        Pipeline de Kedro para regresión de variables numéricas
    """
    return Pipeline(
        [
            # Preparar datos para regresión
            node(
                func=lambda df: prepare_ml_data(df, "regression"),
                inputs="tft_combined_features",
                outputs=["X_regression", "y_regression", "feature_names_regression"],
                name="prepare_regression_data",
                tags=["ml", "regression", "data_preparation"]
            ),
            
            # Entrenar modelo de regresión
            node(
                func=train_regression_model,
                inputs={
                    "X": "X_regression",
                    "y": "y_regression",
                    "params": "params:ml_config"
                },
                outputs="regression_results",
                name="train_regression_model",
                tags=["ml", "regression", "training"]
            ),
            
            # Evaluar importancia de features
            node(
                func=evaluate_feature_importance,
                inputs="regression_results",
                outputs="regression_feature_importance",
                name="evaluate_regression_features",
                tags=["ml", "regression", "feature_analysis"]
            ),
            
            # Guardar modelo de regresión
            node(
                func=lambda results, path: save_ml_models(
                    classification_results=None,
                    regression_results=results,
                    output_path=path
                ),
                inputs={
                    "results": "regression_results",
                    "path": "params:ml_config.models.output_path"
                },
                outputs="regression_model_info",
                name="save_regression_model",
                tags=["ml", "regression", "model_persistence"]
            ),
        ],
        tags=["regression", "ml"]
    )