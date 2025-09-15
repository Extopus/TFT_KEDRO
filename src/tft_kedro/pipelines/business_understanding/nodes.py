"""
Nodos para el pipeline de comprensión del negocio.

Este módulo contiene las funciones principales para analizar y entender
los objetivos del negocio en el contexto del proyecto TFT.
"""

import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def describe_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera un resumen estadístico completo del DataFrame.
    
    Esta función analiza las características básicas de un dataset
    para comprender su estructura y contenido inicial.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        
    Returns:
        Dict[str, Any]: Diccionario con estadísticas descriptivas que incluye:
            - head: Primeras 5 filas del DataFrame
            - info: Información de tipos de datos y memoria
            - describe: Estadísticas descriptivas (mean, std, min, max, etc.)
            - shape: Dimensiones del DataFrame (filas, columnas)
            
    Raises:
        ValueError: Si el DataFrame está vacío
        TypeError: Si el input no es un DataFrame
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> stats = describe_dataset(df)
        >>> print(stats['shape'])
        (3, 2)
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío")
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El input debe ser un pandas DataFrame")
    
    logger.info(f"Analizando dataset con {df.shape[0]} filas y {df.shape[1]} columnas")
    
    try:
        stats = {
            "head": df.head().to_dict(),
            "info": str(df.info()),
            "describe": df.describe().to_dict(),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        
        logger.info("Análisis descriptivo completado exitosamente")
        return stats
        
    except Exception as e:
        logger.error(f"Error al generar estadísticas descriptivas: {str(e)}")
        raise


def analyze_business_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analiza métricas clave del negocio para TFT.
    
    Identifica patrones y métricas relevantes para la comprensión
    del negocio en el contexto de Teamfight Tactics.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de partidas de TFT
        
    Returns:
        Dict[str, Any]: Diccionario con análisis de métricas de negocio:
            - total_games: Número total de partidas
            - avg_placement: Placement promedio
            - rank_distribution: Distribución por rangos
            - key_metrics: Métricas clave por rango
            
    Example:
        >>> business_metrics = analyze_business_metrics(tft_data)
        >>> print(business_metrics['total_games'])
        150000
    """
    logger.info("Iniciando análisis de métricas de negocio")
    
    try:
        # Métricas básicas
        total_games = len(df)
        avg_placement = df['placement'].mean() if 'placement' in df.columns else None
        
        # Distribución por rangos (si existe columna rank)
        rank_distribution = {}
        if 'rank' in df.columns:
            rank_distribution = df['rank'].value_counts().to_dict()
        
        # Métricas clave por rango
        key_metrics = {}
        if 'rank' in df.columns and 'placement' in df.columns:
            key_metrics = df.groupby('rank')['placement'].agg(['mean', 'std', 'count']).to_dict()
        
        analysis = {
            "total_games": total_games,
            "avg_placement": avg_placement,
            "rank_distribution": rank_distribution,
            "key_metrics": key_metrics,
            "data_quality": {
                "missing_values": df.isnull().sum().sum(),
                "duplicate_rows": df.duplicated().sum()
            }
        }
        
        logger.info(f"Análisis de negocio completado: {total_games} partidas analizadas")
        return analysis
        
    except Exception as e:
        logger.error(f"Error en análisis de métricas de negocio: {str(e)}")
        raise
