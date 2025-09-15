"""
Pipeline de Reporting para el proyecto TFT Kedro.

Este pipeline genera reportes finales, dashboards y visualizaciones
del análisis completo de Machine Learning en TFT.
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    generate_executive_summary,
    create_performance_dashboard,
    generate_model_comparison_report,
    create_insights_visualization,
    generate_final_report
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de Reporting para TFT.
    
    Returns:
        Pipeline de Kedro con nodos de reporting
    """
    return Pipeline(
        [
            # Generar resumen ejecutivo
            node(
                func=generate_executive_summary,
                inputs=["ml_insights", "classification_results", "regression_results"],
                outputs="executive_summary",
                name="generate_executive_summary",
                tags=["reporting", "summary", "executive"]
            ),
            
            # Crear dashboard de rendimiento
            node(
                func=create_performance_dashboard,
                inputs=["classification_results", "regression_results", "params:ml_config"],
                outputs="dashboard_info",
                name="create_performance_dashboard",
                tags=["reporting", "dashboard", "visualization"]
            ),
            
            # Generar reporte comparativo de modelos
            node(
                func=generate_model_comparison_report,
                inputs=["classification_results", "regression_results", "params:ml_config"],
                outputs="model_report_info",
                name="generate_model_comparison_report",
                tags=["reporting", "comparison", "models"]
            ),
            
            # Crear visualizaciones de insights
            node(
                func=create_insights_visualization,
                inputs=["ml_insights", "params:ml_config"],
                outputs="insights_viz_info",
                name="create_insights_visualization",
                tags=["reporting", "insights", "visualization"]
            ),
            
            # Generar reporte final consolidado
            node(
                func=generate_final_report,
                inputs=["executive_summary", "dashboard_info", "model_report_info", "insights_viz_info", "params:ml_config"],
                outputs="final_report_info",
                name="generate_final_report",
                tags=["reporting", "final", "consolidated"]
            ),
        ],
        tags="reporting"
    )
