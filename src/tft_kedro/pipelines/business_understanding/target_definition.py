"""
Definición y justificación de targets para Machine Learning en TFT.

Este módulo define los targets principales para el proyecto de análisis
de datos de Teamfight Tactics, con justificaciones basadas en el negocio
y análisis de viabilidad técnica.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def define_ml_targets(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Define los targets principales para Machine Learning basados en el análisis de negocio.
    
    Args:
        df: DataFrame con datos de TFT
        
    Returns:
        Dict con definición de targets y su justificación
    """
    logger.info("Definiendo targets ML para proyecto TFT")
    
    targets = {
        "classification": {
            "name": "rank_prediction",
            "description": "Predicción del rango competitivo del jugador",
            "target_column": "rank",  # Se derivará de los datos
            "classes": ["Challenger", "Grandmaster", "Platinum"],
            "business_justification": {
                "objective": "Identificar patrones que diferencien niveles de juego",
                "value": "Ayudar a jugadores a entender qué los diferencia de otros rangos",
                "applications": [
                    "Sistema de recomendaciones para mejora",
                    "Análisis de brechas competitivas",
                    "Identificación de fortalezas por rango"
                ]
            },
            "technical_feasibility": {
                "data_availability": "Alta - datos disponibles por rango",
                "feature_richness": "Alta - múltiples métricas de juego",
                "class_balance": "Evaluar en análisis exploratorio",
                "complexity": "Media - problema multiclase balanceado"
            }
        },
        
        "regression": {
            "name": "placement_prediction", 
            "description": "Predicción del placement final en partidas",
            "target_column": "placement",
            "range": [1, 8],
            "business_justification": {
                "objective": "Predecir rendimiento basado en composición y estrategia",
                "value": "Optimizar decisiones en tiempo real durante partidas",
                "applications": [
                    "Sistema de recomendaciones de comps",
                    "Análisis de viabilidad de estrategias",
                    "Predicción de éxito de composiciones"
                ]
            },
            "technical_feasibility": {
                "data_availability": "Alta - placement disponible en todos los datos",
                "feature_richness": "Alta - métricas detalladas de partida",
                "target_distribution": "Evaluar en EDA",
                "complexity": "Media - regresión con rango limitado"
            }
        }
    }
    
    # Análisis de viabilidad básico
    feasibility_analysis = analyze_target_feasibility(df, targets)
    targets["feasibility_analysis"] = feasibility_analysis
    
    logger.info(f"Targets definidos: {list(targets.keys())}")
    return targets


def analyze_target_feasibility(df: pd.DataFrame, targets: Dict) -> Dict[str, Any]:
    """
    Analiza la viabilidad técnica de los targets propuestos.
    
    Args:
        df: DataFrame con datos
        targets: Diccionario con definición de targets
        
    Returns:
        Dict con análisis de viabilidad
    """
    analysis = {}
    
    # Análisis de datos disponibles
    analysis["data_overview"] = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    # Análisis de features potenciales
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    analysis["feature_analysis"] = {
        "numeric_features": len(numeric_columns),
        "categorical_features": len(categorical_columns),
        "numeric_columns": numeric_columns[:10],  # Primeros 10
        "categorical_columns": categorical_columns[:5]  # Primeros 5
    }
    
    # Evaluación de viabilidad
    analysis["viability_score"] = calculate_viability_score(df, analysis)
    
    return analysis


def calculate_viability_score(df: pd.DataFrame, analysis: Dict) -> Dict[str, float]:
    """
    Calcula un score de viabilidad para cada target.
    
    Args:
        df: DataFrame con datos
        analysis: Análisis previo
        
    Returns:
        Dict con scores de viabilidad
    """
    scores = {}
    
    # Score basado en cantidad de datos
    data_score = min(1.0, len(df) / 10000)  # Óptimo con 10k+ registros
    
    # Score basado en features disponibles
    feature_score = min(1.0, analysis["feature_analysis"]["numeric_features"] / 20)  # Óptimo con 20+ features
    
    # Score basado en calidad de datos (menos missing values = mejor)
    quality_score = max(0.0, 1.0 - (analysis["data_overview"]["missing_data_percentage"] / 100))
    
    # Score combinado
    overall_score = (data_score * 0.4 + feature_score * 0.3 + quality_score * 0.3)
    
    scores = {
        "data_score": round(data_score, 3),
        "feature_score": round(feature_score, 3), 
        "quality_score": round(quality_score, 3),
        "overall_score": round(overall_score, 3),
        "recommendation": "HIGH" if overall_score > 0.7 else "MEDIUM" if overall_score > 0.5 else "LOW"
    }
    
    return scores


def get_target_engineering_plan(targets: Dict) -> Dict[str, List[str]]:
    """
    Genera un plan de feature engineering específico para cada target.
    
    Args:
        targets: Diccionario con targets definidos
        
    Returns:
        Dict con plan de feature engineering
    """
    plan = {
        "classification_features": [
            "Promedio de gold_left por rango",
            "Desviación estándar de placement por rango", 
            "Ratio de victorias por rango",
            "Tiempo promedio de partida por rango",
            "Eliminaciones promedio por rango"
        ],
        
        "regression_features": [
            "Gold acumulado durante la partida",
            "Nivel máximo alcanzado",
            "Daño total infligido",
            "Número de eliminaciones",
            "Tiempo de supervivencia",
            "Eficiencia de economía (gold por round)"
        ],
        
        "common_features": [
            "Normalización de métricas numéricas",
            "Encoding de variables categóricas",
            "Features de interacción entre variables",
            "Métricas agregadas por jugador"
        ]
    }
    
    return plan


def validate_targets_implementation(df: pd.DataFrame, targets: Dict) -> bool:
    """
    Valida que los targets propuestos sean implementables con los datos disponibles.
    
    Args:
        df: DataFrame con datos
        targets: Targets definidos
        
    Returns:
        bool: True si los targets son implementables
    """
    validation_results = []
    
    # Validar que tenemos suficientes datos
    min_records = 1000
    validation_results.append(len(df) >= min_records)
    
    # Validar que tenemos features suficientes
    min_features = 10
    validation_results.append(len(df.columns) >= min_features)
    
    # Validar que no hay demasiados missing values
    max_missing = 50  # 50%
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    validation_results.append(missing_percentage <= max_missing)
    
    is_valid = all(validation_results)
    
    logger.info(f"Validación de targets: {'PASÓ' if is_valid else 'FALLÓ'}")
    logger.info(f"Registros: {len(df)} (mín: {min_records})")
    logger.info(f"Features: {len(df.columns)} (mín: {min_features})")
    logger.info(f"Missing data: {missing_percentage:.1f}% (máx: {max_missing}%)")
    
    return is_valid
