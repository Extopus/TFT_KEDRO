"""
Nodos para el pipeline de Reporting.

Este módulo contiene las funciones para generar reportes finales,
visualizaciones y dashboards del análisis de TFT.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import os

logger = logging.getLogger(__name__)


def generate_executive_summary(ml_insights: Dict[str, Any], 
                              classification_results: Dict[str, Any],
                              regression_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genera un resumen ejecutivo del análisis completo.
    
    Args:
        ml_insights: Insights generados por el pipeline de ML
        classification_results: Resultados del modelo de clasificación
        regression_results: Resultados del modelo de regresión
        
    Returns:
        Dict con resumen ejecutivo
    """
    logger.info("Generando resumen ejecutivo")
    
    try:
        summary = {
            "project_info": {
                "title": "Análisis de Rendimiento en Teamfight Tactics (TFT)",
                "objective": "Identificar variables clave que impactan el rendimiento competitivo",
                "methodology": "CRISP-DM con Machine Learning",
                "generated_date": datetime.now().isoformat(),
                "scope": "Rangos Challenger, Grandmaster y Platinum"
            },
            
            "key_findings": {
                "classification_performance": {
                    "model": classification_results.get('best_model_name', 'N/A'),
                    "accuracy": classification_results.get('metrics', {}).get('accuracy', 0),
                    "interpretation": "Capacidad de predecir el rango del jugador"
                },
                
                "regression_performance": {
                    "model": regression_results.get('best_model_name', 'N/A'),
                    "r2_score": regression_results.get('metrics', {}).get('r2', 0),
                    "rmse": regression_results.get('metrics', {}).get('rmse', 0),
                    "interpretation": "Capacidad de predecir el placement en partidas"
                },
                
                "critical_variables": ml_insights.get('critical_variables', []),
                "strategic_insights": ml_insights.get('strategic_insights', [])
            },
            
            "business_value": {
                "applications": [
                    "Sistema de recomendaciones para mejora de jugadores",
                    "Análisis de brechas competitivas entre rangos",
                    "Identificación de métricas clave de rendimiento",
                    "Base para estrategias de entrenamiento"
                ],
                "target_audience": [
                    "Jugadores competitivos de TFT",
                    "Entrenadores y coaches",
                    "Desarrolladores del juego",
                    "Analistas de datos gaming"
                ]
            }
        }
        
        logger.info("Resumen ejecutivo generado exitosamente")
        return summary
        
    except Exception as e:
        logger.error(f"Error generando resumen ejecutivo: {str(e)}")
        raise


def create_performance_dashboard(classification_results: Dict[str, Any],
                                regression_results: Dict[str, Any],
                                output_path: str) -> Dict[str, str]:
    """
    Crea un dashboard interactivo de rendimiento de modelos.
    
    Args:
        classification_results: Resultados del modelo de clasificación
        regression_results: Resultados del modelo de regresión
        output_path: Ruta para guardar el dashboard
        
    Returns:
        Dict con rutas de archivos generados
    """
    logger.info("Creando dashboard de rendimiento")
    
    try:
        # Crear subplots para el dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Rendimiento de Clasificación',
                'Rendimiento de Regresión',
                'Importancia de Features - Clasificación',
                'Importancia de Features - Regresión'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Gráfico 1: Métricas de clasificación
        if classification_results:
            clf_metrics = classification_results.get('metrics', {})
            clf_names = ['Accuracy', 'CV Mean', 'CV Std']
            clf_values = [
                clf_metrics.get('accuracy', 0),
                clf_metrics.get('cv_mean', 0),
                clf_metrics.get('cv_std', 0)
            ]
            
            fig.add_trace(
                go.Bar(x=clf_names, y=clf_values, name="Clasificación", 
                      marker_color='lightblue'),
                row=1, col=1
            )
        
        # Gráfico 2: Métricas de regresión
        if regression_results:
            reg_metrics = regression_results.get('metrics', {})
            reg_names = ['R² Score', 'RMSE', 'MAE']
            reg_values = [
                reg_metrics.get('r2', 0),
                reg_metrics.get('rmse', 0),
                reg_metrics.get('mae', 0)
            ]
            
            fig.add_trace(
                go.Bar(x=reg_names, y=reg_values, name="Regresión",
                      marker_color='lightgreen'),
                row=1, col=2
            )
        
        # Gráfico 3: Feature importance clasificación
        if classification_results and classification_results.get('feature_importance'):
            features = [f[0] for f in classification_results['feature_importance'][:10]]
            importance = [f[1] for f in classification_results['feature_importance'][:10]]
            
            fig.add_trace(
                go.Bar(x=importance, y=features, orientation='h', name="Features Clasificación",
                      marker_color='lightcoral'),
                row=2, col=1
            )
        
        # Gráfico 4: Feature importance regresión
        if regression_results and regression_results.get('feature_importance'):
            features = [f[0] for f in regression_results['feature_importance'][:10]]
            importance = [f[1] for f in regression_results['feature_importance'][:10]]
            
            fig.add_trace(
                go.Bar(x=importance, y=features, orientation='h', name="Features Regresión",
                      marker_color='lightyellow'),
                row=2, col=2
            )
        
        # Configurar layout
        fig.update_layout(
            title_text="Dashboard de Rendimiento - Modelos TFT",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Guardar dashboard
        output_dir = output_path.get('output_path', 'data/reports')
        dashboard_path = os.path.join(output_dir, "performance_dashboard.html")
        fig.write_html(dashboard_path)
        
        logger.info(f"Dashboard guardado en: {dashboard_path}")
        
        return {
            "dashboard_path": dashboard_path,
            "dashboard_type": "interactive_html"
        }
        
    except Exception as e:
        logger.error(f"Error creando dashboard: {str(e)}")
        raise


def generate_model_comparison_report(classification_results: Dict[str, Any],
                                   regression_results: Dict[str, Any],
                                   output_path: str) -> Dict[str, str]:
    """
    Genera un reporte comparativo de modelos.
    
    Args:
        classification_results: Resultados del modelo de clasificación
        regression_results: Resultados del modelo de regresión
        output_path: Ruta para guardar el reporte
        
    Returns:
        Dict con rutas de archivos generados
    """
    logger.info("Generando reporte comparativo de modelos")
    
    try:
        # Crear reporte en markdown
        report_content = []
        
        report_content.append("# Reporte Comparativo de Modelos TFT")
        report_content.append(f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Sección de clasificación
        report_content.append("## Modelos de Clasificación")
        report_content.append("")
        
        if classification_results:
            clf_metrics = classification_results.get('metrics', {})
            report_content.append(f"**Mejor modelo:** {classification_results.get('best_model_name', 'N/A')}")
            report_content.append(f"**Accuracy:** {clf_metrics.get('accuracy', 0):.4f}")
            report_content.append(f"**CV Score:** {clf_metrics.get('cv_mean', 0):.4f} ± {clf_metrics.get('cv_std', 0):.4f}")
            report_content.append("")
            
            # Feature importance
            if classification_results.get('feature_importance'):
                report_content.append("### Top 5 Features más Importantes:")
                for i, (feature, importance) in enumerate(classification_results['feature_importance'][:5], 1):
                    report_content.append(f"{i}. **{feature}**: {importance:.4f}")
                report_content.append("")
        
        # Sección de regresión
        report_content.append("## Modelos de Regresión")
        report_content.append("")
        
        if regression_results:
            reg_metrics = regression_results.get('metrics', {})
            report_content.append(f"**Mejor modelo:** {regression_results.get('best_model_name', 'N/A')}")
            report_content.append(f"**R² Score:** {reg_metrics.get('r2', 0):.4f}")
            report_content.append(f"**RMSE:** {reg_metrics.get('rmse', 0):.4f}")
            report_content.append(f"**MAE:** {reg_metrics.get('mae', 0):.4f}")
            report_content.append("")
            
            # Feature importance
            if regression_results.get('feature_importance'):
                report_content.append("### Top 5 Features más Importantes:")
                for i, (feature, importance) in enumerate(regression_results['feature_importance'][:5], 1):
                    report_content.append(f"{i}. **{feature}**: {importance:.4f}")
                report_content.append("")
        
        # Conclusiones
        report_content.append("## Conclusiones")
        report_content.append("")
        report_content.append("- Los modelos desarrollados proporcionan insights valiosos sobre el rendimiento en TFT")
        report_content.append("- Las variables identificadas como importantes pueden guiar estrategias de mejora")
        report_content.append("- Los modelos pueden ser utilizados para predicciones y análisis de patrones")
        report_content.append("")
        
        # Guardar reporte
        output_dir = output_path.get('output_path', 'data/reports')
        report_path = os.path.join(output_dir, "model_comparison_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"Reporte guardado en: {report_path}")
        
        return {
            "report_path": report_path,
            "report_type": "markdown"
        }
        
    except Exception as e:
        logger.error(f"Error generando reporte comparativo: {str(e)}")
        raise


def create_insights_visualization(ml_insights: Dict[str, Any], 
                                 output_path: str) -> Dict[str, str]:
    """
    Crea visualizaciones de insights estratégicos.
    
    Args:
        ml_insights: Insights generados por el pipeline de ML
        output_path: Ruta para guardar las visualizaciones
        
    Returns:
        Dict con rutas de archivos generados
    """
    logger.info("Creando visualizaciones de insights")
    
    try:
        # Crear figura para insights
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Variables Críticas Identificadas',
                'Distribución de Insights por Categoría'
            ),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Gráfico 1: Variables críticas
        critical_vars = ml_insights.get('critical_variables', [])
        if critical_vars:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(critical_vars))),
                    y=[1] * len(critical_vars),  # Todas tienen importancia igual para visualización
                    text=critical_vars,
                    textposition='auto',
                    name="Variables Críticas",
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
        
        # Gráfico 2: Categorías de insights
        strategic_insights = ml_insights.get('strategic_insights', [])
        if strategic_insights:
            # Categorizar insights por palabras clave
            categories = {
                'Modelo Efectivo': sum(1 for insight in strategic_insights if 'efectivo' in insight.lower()),
                'Variables Críticas': sum(1 for insight in strategic_insights if 'variables' in insight.lower()),
                'Predicción': sum(1 for insight in strategic_insights if 'predecir' in insight.lower()),
                'Otros': len(strategic_insights) - sum([
                    sum(1 for insight in strategic_insights if 'efectivo' in insight.lower()),
                    sum(1 for insight in strategic_insights if 'variables' in insight.lower()),
                    sum(1 for insight in strategic_insights if 'predecir' in insight.lower())
                ])
            }
            
            # Filtrar categorías con valores > 0
            categories = {k: v for k, v in categories.items() if v > 0}
            
            if categories:
                fig.add_trace(
                    go.Pie(
                        labels=list(categories.keys()),
                        values=list(categories.values()),
                        name="Categorías de Insights"
                    ),
                    row=1, col=2
                )
        
        # Configurar layout
        fig.update_layout(
            title_text="Insights Estratégicos del Análisis TFT",
            title_x=0.5,
            height=600,
            showlegend=True
        )
        
        # Guardar visualización
        insights_path = os.path.join(output_path, "insights_visualization.html")
        fig.write_html(insights_path)
        
        logger.info(f"Visualización de insights guardada en: {insights_path}")
        
        return {
            "insights_visualization_path": insights_path,
            "visualization_type": "interactive_html"
        }
        
    except Exception as e:
        logger.error(f"Error creando visualización de insights: {str(e)}")
        raise


def generate_final_report(executive_summary: Dict[str, Any],
                         dashboard_info: Dict[str, str],
                         model_report_info: Dict[str, str],
                         insights_viz_info: Dict[str, str],
                         output_path: str) -> Dict[str, Any]:
    """
    Genera el reporte final consolidado del proyecto.
    
    Args:
        executive_summary: Resumen ejecutivo
        dashboard_info: Información del dashboard
        model_report_info: Información del reporte de modelos
        insights_viz_info: Información de visualizaciones
        output_path: Ruta para guardar el reporte final
        
    Returns:
        Dict con información del reporte final
    """
    logger.info("Generando reporte final consolidado")
    
    try:
        # Crear reporte final en JSON
        final_report = {
            "project_summary": executive_summary,
            "generated_files": {
                "dashboard": dashboard_info,
                "model_comparison": model_report_info,
                "insights_visualization": insights_viz_info
            },
            "project_status": {
                "completion_percentage": 100,
                "pipelines_completed": [
                    "business_understanding",
                    "data_cleaning", 
                    "feature_engineering",
                    "data_science",
                    "reporting"
                ],
                "methodology": "CRISP-DM",
                "total_analysis_time": datetime.now().isoformat()
            },
            "next_steps": [
                "Implementar modelos en producción",
                "Crear dashboard interactivo en tiempo real",
                "Desarrollar API para predicciones",
                "Expandir análisis a más rangos competitivos"
            ]
        }
        
        # Guardar reporte final
        final_report_path = os.path.join(output_path, "final_project_report.json")
        with open(final_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reporte final guardado en: {final_report_path}")
        
        return {
            "final_report_path": final_report_path,
            "report_type": "consolidated_json",
            "project_completion": "100%"
        }
        
    except Exception as e:
        logger.error(f"Error generando reporte final: {str(e)}")
        raise
