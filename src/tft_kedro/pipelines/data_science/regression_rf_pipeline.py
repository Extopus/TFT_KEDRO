"""
Pipeline de Regresión (RandomForest) para TFT Kedro.

Este pipeline entrenará únicamente un RandomForestRegressor.
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_ml_data,
    train_regression_random_forest,
    evaluate_feature_importance,
    save_ml_models
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=lambda df: prepare_ml_data(df, "regression"),
            inputs="tft_combined_features",
            outputs=["X_regression_rf", "y_regression_rf", "feature_names_regression_rf"],
            name="prepare_regression_data_rf",
            tags=["ml", "regression", "data_preparation"]
        ),
        node(
            func=train_regression_random_forest,
            inputs={
                "X": "X_regression_rf",
                "y": "y_regression_rf",
                "params": "params:ml_config"
            },
            outputs="regression_results_rf",
            name="train_regression_random_forest",
            tags=["ml", "regression", "training"]
        ),
        node(
            func=evaluate_feature_importance,
            inputs="regression_results_rf",
            outputs="regression_feature_importance_rf",
            name="evaluate_regression_features_rf",
            tags=["ml", "regression", "feature_analysis"]
        ),
        node(
            func=lambda results, path: save_ml_models(
                classification_results=None,
                regression_results=results,
                output_path=path
            ),
            inputs={
                "results": "regression_results_rf",
                "path": "params:ml_config.models.output_path"
            },
            outputs="regression_model_info_rf",
            name="save_regression_model_rf",
            tags=["ml", "regression", "model_persistence"]
        )
    ], tags=["regression_rf", "ml"])