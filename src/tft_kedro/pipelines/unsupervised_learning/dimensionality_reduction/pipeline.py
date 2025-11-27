"""
Pipeline de Reducción de Dimensionalidad.

Este pipeline implementa:
- PCA (Análisis de Componentes Principales)
- t-SNE
- UMAP (opcional)
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_data_for_dimensionality_reduction,
    apply_pca,
    apply_tsne,
    apply_umap,
    analyze_pca_components
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de reducción de dimensionalidad.
    
    Returns:
        Pipeline de Kedro con nodos de reducción dimensional
    """
    return Pipeline(
        [
            # Preparar datos para reducción dimensional
            node(
                func=prepare_data_for_dimensionality_reduction,
                inputs=["tft_combined_features", "params:unsupervised_config"],
                outputs=["dim_reduction_data_scaled", "dim_reduction_scaler", "dim_reduction_feature_names"],
                name="prepare_dim_reduction_data",
                tags=["unsupervised", "dimensionality_reduction", "data_preparation"]
            ),
            
            # Aplicar PCA
            node(
                func=apply_pca,
                inputs=["dim_reduction_data_scaled", "params:unsupervised_config"],
                outputs="pca_results",
                name="apply_pca",
                tags=["unsupervised", "dimensionality_reduction", "pca"]
            ),
            
            # Analizar componentes PCA
            node(
                func=analyze_pca_components,
                inputs=["pca_results", "dim_reduction_feature_names"],
                outputs="pca_analysis",
                name="analyze_pca_components",
                tags=["unsupervised", "dimensionality_reduction", "pca", "analysis"]
            ),
            
            # Aplicar t-SNE (usando datos originales o PCA reducido)
            node(
                func=apply_tsne,
                inputs=["dim_reduction_data_scaled", "params:unsupervised_config"],
                outputs="tsne_results",
                name="apply_tsne",
                tags=["unsupervised", "dimensionality_reduction", "tsne"]
            ),
            
            # Aplicar UMAP (opcional, solo si está disponible)
            # node(
            #     func=apply_umap,
            #     inputs=["dim_reduction_data_scaled", "params:unsupervised_config"],
            #     outputs="umap_results",
            #     name="apply_umap",
            #     tags=["unsupervised", "dimensionality_reduction", "umap"]
            # ),
        ],
        tags="dimensionality_reduction"
    )

