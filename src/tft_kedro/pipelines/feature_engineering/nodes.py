import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features derivadas al dataset.
    
    Args:
        df: DataFrame con datos originales
        
    Returns:
        DataFrame con features adicionales
    """
    df = df.copy()
    df['gameDuration_min'] = df['gameDuration'] / 60
    # Agrega aquí más transformaciones si lo deseas
    return df


def combine_features(challenger_features: pd.DataFrame, 
                    platinum_features: pd.DataFrame, 
                    grandmaster_features: pd.DataFrame) -> pd.DataFrame:
    """
    Combina los features de los tres rangos en un solo dataset.
    
    Args:
        challenger_features: Features del rango Challenger
        platinum_features: Features del rango Platinum
        grandmaster_features: Features del rango Grandmaster
        
    Returns:
        DataFrame combinado con columna 'rank' agregada
    """
    logger.info("Combinando features de los tres rangos...")
    
    # Agregar columna de rango a cada dataset
    challenger_features = challenger_features.copy()
    challenger_features['rank'] = 'Challenger'
    
    platinum_features = platinum_features.copy()
    platinum_features['rank'] = 'Platinum'
    
    grandmaster_features = grandmaster_features.copy()
    grandmaster_features['rank'] = 'Grandmaster'
    
    # Combinar todos los datasets
    combined_df = pd.concat([
        challenger_features,
        platinum_features,
        grandmaster_features
    ], ignore_index=True)
    
    logger.info(f"Dataset combinado creado: {combined_df.shape}")
    logger.info(f"Distribución por rango: {combined_df['rank'].value_counts().to_dict()}")
    
    return combined_df
