"""
Pipeline de Clasificación (RandomForest) para TFT Kedro.

Este pipeline entrenará únicamente un RandomForestClassifier.
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_ml_data,
    train_classification_random_forest,
    evaluate_feature_importance,
    save_ml_models
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=lambda df: prepare_ml_data(df, "classification"),
            inputs="tft_combined_features",
            outputs=["X_classification_rf", "y_classification_rf", "feature_names_classification_rf"],
            name="prepare_classification_data_rf",
            tags=["ml", "classification", "data_preparation"]
        ),
        node(
            func=train_classification_random_forest,
            inputs={
                "X": "X_classification_rf",
                "y": "y_classification_rf",
                "params": "params:ml_config"
            },
            outputs="classification_results_rf",
            name="train_classification_random_forest",
            tags=["ml", "classification", "training"]
        ),
        node(
            func=evaluate_feature_importance,
            inputs="classification_results_rf",
            outputs="classification_feature_importance_rf",
            name="evaluate_classification_features_rf",
            tags=["ml", "classification", "feature_analysis"]
        ),
        node(
            func=lambda results, path: save_ml_models(
                classification_results=results,
                regression_results=None,
                output_path=path
            ),
            inputs={
                "results": "classification_results_rf",
                "path": "params:ml_config.models.output_path"
            },
            outputs="classification_model_info_rf",
            name="save_classification_model_rf",
            tags=["ml", "classification", "model_persistence"]
        )
    ], tags=["classification_rf", "ml"])