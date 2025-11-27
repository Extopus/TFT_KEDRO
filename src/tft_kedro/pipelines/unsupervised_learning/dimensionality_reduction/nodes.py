"""
Nodos para el pipeline de Reducción de Dimensionalidad.

Implementa:
- PCA (Análisis de Componentes Principales)
- t-SNE
- UMAP (opcional)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Intentar importar UMAP (opcional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP no está disponible. Instalar con: pip install umap-learn")


def prepare_data_for_dimensionality_reduction(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Prepara los datos para reducción de dimensionalidad.
    
    Args:
        df: DataFrame con features
        params: Parámetros de configuración
        
    Returns:
        Tuple con (datos escalados, scaler, nombres de features)
    """
    logger.info("Preparando datos para reducción de dimensionalidad...")
    
    # Obtener columnas numéricas (excluir IDs y targets)
    exclude_cols = params.get('exclude_columns', ['gameId', 'rank', 'Ranked'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Seleccionar y limpiar datos
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    
    logger.info(f"Datos preparados: {X_scaled_df.shape}")
    logger.info(f"Features utilizadas: {len(feature_cols)}")
    
    return X_scaled_df, scaler, feature_cols


def apply_pca(
    X: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aplica PCA (Análisis de Componentes Principales).
    
    Args:
        X: Datos escalados
        params: Parámetros de configuración
        
    Returns:
        Diccionario con modelo PCA, componentes, varianza explicada, loadings
    """
    logger.info("Aplicando PCA...")
    
    n_components = params.get('pca_n_components', None)  # None = todos los componentes
    random_state = params.get('random_state', 42)
    
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    # Varianza explicada
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Número de componentes para explicar 95% de varianza
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    # Loadings (coeficientes de los componentes)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=X.columns
    )
    
    # Crear DataFrame con componentes principales
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=X.index
    )
    
    results = {
        'algorithm': 'PCA',
        'model': pca,
        'transformed_data': pca_df,
        'n_components': pca.n_components_,
        'explained_variance_ratio': explained_variance.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'n_components_95_variance': int(n_components_95),
        'loadings': loadings.to_dict(),
        'feature_names': X.columns.tolist(),
        'mean': pca.mean_.tolist(),
        'components': pca.components_.tolist()
    }
    
    logger.info(f"PCA completado: {pca.n_components_} componentes")
    logger.info(f"Varianza explicada por PC1: {explained_variance[0]:.4f}")
    logger.info(f"Componentes para 95% varianza: {n_components_95}")
    
    return results


def apply_tsne(
    X: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aplica t-SNE para visualización.
    
    Args:
        X: Datos escalados (puede ser datos originales o PCA reducido)
        params: Parámetros de configuración
        
    Returns:
        Diccionario con modelo t-SNE y datos transformados
    """
    logger.info("Aplicando t-SNE...")
    
    n_components = params.get('tsne_n_components', 2)
    perplexity = params.get('tsne_perplexity', 30)
    random_state = params.get('random_state', 42)
    n_iter = params.get('tsne_n_iter', 1000)
    
    # Para datasets grandes, usar submuestra o PCA previo
    max_samples = params.get('tsne_max_samples', 10000)
    if len(X) > max_samples:
        logger.info(f"Dataset grande ({len(X)} muestras). Usando submuestra de {max_samples} para t-SNE...")
        sample_indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_indices] if isinstance(X, pd.DataFrame) else X[sample_indices]
        use_sample = True
    else:
        X_sample = X
        sample_indices = None
        use_sample = False
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=n_iter,
        verbose=1
    )
    
    X_tsne = tsne.fit_transform(X_sample)
    
    # Crear DataFrame con componentes t-SNE
    tsne_df = pd.DataFrame(
        X_tsne,
        columns=[f't-SNE_{i+1}' for i in range(n_components)],
        index=X_sample.index if isinstance(X_sample, pd.DataFrame) else range(len(X_sample))
    )
    
    results = {
        'algorithm': 't-SNE',
        'model': tsne,
        'transformed_data': tsne_df,
        'n_components': n_components,
        'perplexity': perplexity,
        'kl_divergence': float(tsne.kl_divergence_),
        'n_iter': tsne.n_iter_,
        'used_sample': use_sample,
        'sample_indices': sample_indices.tolist() if sample_indices is not None else None
    }
    
    logger.info(f"t-SNE completado: {n_components} componentes, KL divergence: {tsne.kl_divergence_:.4f}")
    
    return results


def apply_umap(
    X: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aplica UMAP para reducción de dimensionalidad.
    
    Args:
        X: Datos escalados
        params: Parámetros de configuración
        
    Returns:
        Diccionario con modelo UMAP y datos transformados
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP no está disponible. Instalar con: pip install umap-learn")
    
    logger.info("Aplicando UMAP...")
    
    n_components = params.get('umap_n_components', 2)
    n_neighbors = params.get('umap_n_neighbors', 15)
    min_dist = params.get('umap_min_dist', 0.1)
    random_state = params.get('random_state', 42)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    X_umap = reducer.fit_transform(X)
    
    # Crear DataFrame con componentes UMAP
    umap_df = pd.DataFrame(
        X_umap,
        columns=[f'UMAP_{i+1}' for i in range(n_components)],
        index=X.index
    )
    
    results = {
        'algorithm': 'UMAP',
        'model': reducer,
        'transformed_data': umap_df,
        'n_components': n_components,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist
    }
    
    logger.info(f"UMAP completado: {n_components} componentes")
    
    return results


def analyze_pca_components(
    pca_results: Dict[str, Any],
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Analiza los componentes principales de PCA.
    
    Args:
        pca_results: Resultados de PCA
        feature_names: Nombres de las features
        
    Returns:
        Diccionario con análisis de componentes
    """
    logger.info("Analizando componentes principales de PCA...")
    
    loadings = pd.DataFrame(pca_results['loadings'])
    
    # Top features por componente principal
    top_features_per_pc = {}
    n_top = 5
    
    for pc in loadings.columns:
        top_features = loadings[pc].abs().nlargest(n_top)
        top_features_per_pc[pc] = {
            'features': top_features.index.tolist(),
            'loadings': top_features.values.tolist()
        }
    
    # Interpretación de PC1
    pc1_features = loadings['PC1'].abs().nlargest(3)
    pc1_interpretation = {
        'top_features': pc1_features.index.tolist(),
        'loadings': pc1_features.values.tolist(),
        'direction': 'positive' if loadings['PC1'].sum() > 0 else 'negative'
    }
    
    results = {
        'top_features_per_component': top_features_per_pc,
        'pc1_interpretation': pc1_interpretation,
        'explained_variance_pc1': pca_results['explained_variance_ratio'][0],
        'explained_variance_pc2': pca_results['explained_variance_ratio'][1] if len(pca_results['explained_variance_ratio']) > 1 else None,
        'cumulative_variance_2_components': pca_results['cumulative_variance'][1] if len(pca_results['cumulative_variance']) > 1 else None
    }
    
    logger.info(f"Análisis completado: PC1 explica {results['explained_variance_pc1']:.2%} de varianza")
    
    return results

