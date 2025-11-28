"""
Pipeline principal de Aprendizaje No Supervisado.

Integra clustering y reducción de dimensionalidad.
"""

from kedro.pipeline import Pipeline
from .clustering.pipeline import create_pipeline as create_clustering_pipeline
from .dimensionality_reduction.pipeline import create_pipeline as create_dim_reduction_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline completo de aprendizaje no supervisado.
    
    Returns:
        Pipeline de Kedro que integra clustering y reducción dimensional
    """
    clustering_pipeline = create_clustering_pipeline()
    dim_reduction_pipeline = create_dim_reduction_pipeline()
    
    # Combinar pipelines
    return clustering_pipeline + dim_reduction_pipeline


