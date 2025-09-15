"""
Nodos para el pipeline de limpieza de datos.

Este módulo contiene las funciones para limpiar, validar y preprocesar
los datos de TFT según las mejores prácticas de data engineering.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def clean_csv_data(df: pd.DataFrame, 
                  remove_missing: bool = True,
                  remove_duplicates: bool = True,
                  reset_index: bool = True) -> pd.DataFrame:
    """
    Limpia y preprocesa datos CSV de TFT.
    
    Aplica técnicas de limpieza de datos incluyendo manejo de valores
    faltantes, eliminación de duplicados y normalización del índice.
    
    Args:
        df (pd.DataFrame): DataFrame con datos crudos de TFT
        remove_missing (bool, optional): Si eliminar filas con valores nulos. 
                                        Defaults to True.
        remove_duplicates (bool, optional): Si eliminar filas duplicadas. 
                                           Defaults to True.
        reset_index (bool, optional): Si resetear el índice del DataFrame. 
                                     Defaults to True.
    
    Returns:
        pd.DataFrame: DataFrame limpio y preprocesado
        
    Raises:
        ValueError: Si el DataFrame está vacío
        TypeError: Si el input no es un DataFrame
        
    Example:
        >>> raw_data = pd.DataFrame({'A': [1, None, 1], 'B': [2, 3, 2]})
        >>> clean_data = clean_csv_data(raw_data)
        >>> print(len(clean_data))
        1
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío")
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El input debe ser un pandas DataFrame")
    
    logger.info(f"Iniciando limpieza de datos: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    original_shape = df.shape
    
    try:
        # Manejo de valores faltantes
        if remove_missing:
            missing_before = df.isnull().sum().sum()
            df = df.dropna()
            missing_removed = missing_before - df.isnull().sum().sum()
            logger.info(f"Eliminadas {missing_removed} filas con valores faltantes")
        
        # Eliminación de duplicados
        if remove_duplicates:
            duplicates_before = df.duplicated().sum()
            df = df.drop_duplicates()
            duplicates_removed = duplicates_before - df.duplicated().sum()
            logger.info(f"Eliminadas {duplicates_removed} filas duplicadas")
        
        # Resetear índice
        if reset_index:
            df = df.reset_index(drop=True)
        
        final_shape = df.shape
        logger.info(f"Limpieza completada: {original_shape} → {final_shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error en limpieza de datos: {str(e)}")
        raise


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida la calidad de los datos después de la limpieza.
    
    Realiza verificaciones de calidad incluyendo distribución de datos,
    detección de outliers y análisis de consistencia.
    
    Args:
        df (pd.DataFrame): DataFrame limpio para validar
        
    Returns:
        Dict[str, Any]: Diccionario con métricas de calidad:
            - missing_values: Conteo de valores faltantes por columna
            - duplicates: Número de filas duplicadas
            - outliers: Detección básica de outliers
            - data_types: Análisis de tipos de datos
            - quality_score: Score general de calidad (0-1)
            
    Example:
        >>> quality_report = validate_data_quality(clean_df)
        >>> print(quality_report['quality_score'])
        0.95
    """
    logger.info("Iniciando validación de calidad de datos")
    
    try:
        # Análisis de valores faltantes
        missing_analysis = {
            "total_missing": df.isnull().sum().sum(),
            "missing_by_column": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Análisis de duplicados
        duplicate_analysis = {
            "total_duplicates": df.duplicated().sum(),
            "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
        }
        
        # Análisis de tipos de datos
        dtype_analysis = {
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist()
        }
        
        # Detección básica de outliers (para columnas numéricas)
        outlier_analysis = {}
        numeric_cols = dtype_analysis["numeric_columns"]
        if numeric_cols:
            for col in numeric_cols[:5]:  # Solo primeras 5 columnas numéricas
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_analysis[col] = len(outliers)
        
        # Cálculo de score de calidad
        quality_factors = [
            1.0 - (missing_analysis["missing_percentage"] / 100),  # Menos missing = mejor
            1.0 - (duplicate_analysis["duplicate_percentage"] / 100),  # Menos duplicados = mejor
            0.9 if len(dtype_analysis["numeric_columns"]) > 0 else 0.5,  # Tiene datos numéricos
        ]
        quality_score = np.mean(quality_factors)
        
        quality_report = {
            "missing_values": missing_analysis,
            "duplicates": duplicate_analysis,
            "data_types": dtype_analysis,
            "outliers": outlier_analysis,
            "quality_score": round(quality_score, 3),
            "recommendation": "EXCELLENT" if quality_score > 0.9 else "GOOD" if quality_score > 0.7 else "NEEDS_IMPROVEMENT"
        }
        
        logger.info(f"Validación completada - Score de calidad: {quality_score:.3f}")
        return quality_report
        
    except Exception as e:
        logger.error(f"Error en validación de calidad: {str(e)}")
        raise


def handle_outliers(df: pd.DataFrame, 
                   method: str = "iqr",
                   columns: Optional[list] = None) -> pd.DataFrame:
    """
    Maneja outliers en columnas numéricas del dataset.
    
    Aplica diferentes estrategias para el tratamiento de outliers
    según el método especificado.
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        method (str, optional): Método para manejo de outliers. 
                               Opciones: "iqr", "zscore", "clip". Defaults to "iqr".
        columns (Optional[list], optional): Lista de columnas a procesar. 
                                           Si None, procesa todas las numéricas. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame con outliers tratados
        
    Raises:
        ValueError: Si el método no es válido
        
    Example:
        >>> df_no_outliers = handle_outliers(df, method="iqr")
    """
    logger.info(f"Aplicando método '{method}' para manejo de outliers")
    
    valid_methods = ["iqr", "zscore", "clip"]
    if method not in valid_methods:
        raise ValueError(f"Método '{method}' no válido. Opciones: {valid_methods}")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_processed = df.copy()
    
    try:
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Columna '{col}' no encontrada, omitiendo")
                continue
                
            original_outliers = 0
            
            if method == "iqr":
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                original_outliers = len(df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)])
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
                original_outliers = len(df_processed[z_scores > 3])
                df_processed = df_processed[z_scores <= 3]
                
            elif method == "clip":
                lower_bound = df_processed[col].quantile(0.05)
                upper_bound = df_processed[col].quantile(0.95)
                original_outliers = len(df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)])
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            
            logger.info(f"Columna '{col}': {original_outliers} outliers tratados")
        
        logger.info("Manejo de outliers completado")
        return df_processed
        
    except Exception as e:
        logger.error(f"Error en manejo de outliers: {str(e)}")
        raise
