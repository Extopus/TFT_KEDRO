"""
Pipeline de Clasificación para TFT Kedro.

Este pipeline implementa modelos de clasificación para predecir rangos competitivos
(Challenger/Grandmaster/Platinum).
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_ml_data,
    train_classification_model,
    evaluate_feature_importance,
    save_ml_models
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de clasificación.
    
    Returns:
        Pipeline de Kedro para clasificación de rangos
    """
    return Pipeline(
        [
            # Preparar datos para clasificación
            node(
                func=lambda df: prepare_ml_data(df, "classification"),
                inputs="tft_combined_features",
                outputs=["X_classification", "y_classification", "feature_names_classification"],
                name="prepare_classification_data",
                tags=["ml", "classification", "data_preparation"]
            ),
            
            # Entrenar modelo de clasificación
            node(
                func=train_classification_model,
                inputs={
                    "X": "X_classification",
                    "y": "y_classification",
                    "params": "params:ml_config"
                },
                outputs="classification_results",
                name="train_classification_model",
                tags=["ml", "classification", "training"]
            ),
            
            # Evaluar importancia de features
            node(
                func=evaluate_feature_importance,
                inputs="classification_results",
                outputs="classification_feature_importance",
                name="evaluate_classification_features",
                tags=["ml", "classification", "feature_analysis"]
            ),
            
            # Guardar modelo de clasificación
            node(
                func=lambda results, path: save_ml_models(
                    classification_results=results,
                    regression_results=None,
                    output_path=path
                ),
                inputs={
                    "results": "classification_results",
                    "path": "params:ml_config.models.output_path"
                },
                outputs="classification_model_info",
                name="save_classification_model",
                tags=["ml", "classification", "model_persistence"]
            ),
        ],
        tags=["classification", "ml"]
    )