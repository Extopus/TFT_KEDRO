"""
Nodos para el pipeline de Data Science.

Este módulo contiene las funciones para implementar modelos de Machine Learning
para clasificación de rangos y regresión de placement en TFT.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import joblib
import json

logger = logging.getLogger(__name__)


def prepare_ml_data(df: pd.DataFrame, target_type: str = "classification") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepara los datos para machine learning.
    
    Args:
        df: DataFrame con datos de TFT
        target_type: Tipo de target ("classification" o "regression")
        
    Returns:
        Tuple con (X, y, feature_names)
    """
    logger.info(f"Preparando datos para {target_type}")
    
    try:
        # Eliminar columnas no relevantes para ML
        ml_df = df.copy()
        
        # Identificar variables numéricas como features
        numeric_features = ml_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remover columnas que no son features útiles
        exclude_cols = ['rank', 'placement'] if 'placement' in ml_df.columns else ['rank']
        feature_columns = [col for col in numeric_features if col not in exclude_cols]
        
        # Preparar features (X)
        X = ml_df[feature_columns].fillna(ml_df[feature_columns].median())
        
        # Preparar target (y) según el tipo
        if target_type == "classification":
            # Clasificación: predecir rango
            y = ml_df['rank']
            logger.info(f"Target de clasificación: {y.value_counts().to_dict()}")
        else:
            # Regresión: predecir placement (usar columna 'Ranked')
            if 'Ranked' in ml_df.columns:
                y = ml_df['Ranked']
                logger.info(f"Target de regresión: rango {y.min()}-{y.max()}, media {y.mean():.2f}")
            else:
                raise ValueError("Columna 'Ranked' no encontrada para regresión")
        
        logger.info(f"Features preparadas: {len(feature_columns)} variables")
        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        return X, y, feature_columns
        
    except Exception as e:
        logger.error(f"Error preparando datos ML: {str(e)}")
        raise


def train_classification_model(X: pd.DataFrame, y: pd.Series, params: Dict) -> Dict[str, Any]:
    """
    Entrena un modelo de clasificación para predecir el rango del jugador.
    
    Args:
        X: Features para entrenamiento
        y: Target de clasificación (rank)
        params: Parámetros de configuración
        
    Returns:
        Dict con modelo entrenado, métricas y resultados
    """
    logger.info("Iniciando entrenamiento de modelo de clasificación")
    
    try:
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params.get('test_size', 0.2), 
            random_state=params.get('random_state', 42), stratify=y
        )
        
        # Escalado de features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar múltiples modelos
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Entrenando modelo: {name}")
            
            # Usar datos escalados para LogisticRegression, originales para RandomForest
            if name == 'LogisticRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'scaler': scaler if name == 'LogisticRegression' else None,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_actual': y_test,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Seleccionar mejor modelo
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_model = results[best_model_name]
        
        logger.info(f"Mejor modelo: {best_model_name} (Accuracy: {best_model['accuracy']:.4f})")
        
        return {
            'best_model_name': best_model_name,
            'best_model': best_model['model'],
            'scaler': best_model['scaler'],
            'results': results,
            'feature_names': X.columns.tolist(),
            'metrics': {
                'accuracy': best_model['accuracy'],
                'cv_mean': best_model['cv_mean'],
                'cv_std': best_model['cv_std']
            }
        }
        
    except Exception as e:
        logger.error(f"Error entrenando modelo de clasificación: {str(e)}")
        raise


def train_regression_model(X: pd.DataFrame, y: pd.Series, params: Dict) -> Dict[str, Any]:
    """
    Entrena un modelo de regresión para predecir el placement.
    
    Args:
        X: Features para entrenamiento
        y: Target de regresión (placement)
        params: Parámetros de configuración
        
    Returns:
        Dict con modelo entrenado, métricas y resultados
    """
    logger.info("Iniciando entrenamiento de modelo de regresión")
    
    try:
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params.get('test_size', 0.2), 
            random_state=params.get('random_state', 42)
        )
        
        # Escalado de features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar múltiples modelos
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'LinearRegression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Entrenando modelo: {name}")
            
            # Usar datos escalados para LinearRegression, originales para RandomForest
            if name == 'LinearRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'scaler': scaler if name == 'LinearRegression' else None,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'test_actual': y_test
            }
            
            logger.info(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Seleccionar mejor modelo (mayor R²)
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_model = results[best_model_name]
        
        logger.info(f"Mejor modelo: {best_model_name} (R²: {best_model['r2']:.4f})")
        
        return {
            'best_model_name': best_model_name,
            'best_model': best_model['model'],
            'scaler': best_model['scaler'],
            'results': results,
            'feature_names': X.columns.tolist(),
            'metrics': {
                'r2': best_model['r2'],
                'rmse': best_model['rmse'],
                'mae': best_model['mae'],
                'cv_mean': best_model['cv_mean'],
                'cv_std': best_model['cv_std']
            }
        }
        
    except Exception as e:
        logger.error(f"Error entrenando modelo de regresión: {str(e)}")
        raise


def evaluate_feature_importance(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evalúa la importancia de las features en el mejor modelo.
    
    Args:
        model_results: Resultados del entrenamiento de modelos
        
    Returns:
        Dict con análisis de importancia de features
    """
    logger.info("Evaluando importancia de features")
    
    try:
        feature_names = model_results['feature_names']
        best_model = model_results['best_model']
        
        # Obtener importancia de features
        if hasattr(best_model, 'feature_importances_'):
            # Para RandomForest
            importances = best_model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Feature importance (RandomForest):")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")
                
        elif hasattr(best_model, 'coef_'):
            # Para modelos lineales
            coefs = best_model.coef_
            if len(coefs.shape) > 1:
                # Para clasificación multiclase
                coefs = np.mean(np.abs(coefs), axis=0)
            
            feature_importance = list(zip(feature_names, np.abs(coefs)))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Feature importance (Linear model):")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")
        else:
            logger.warning("Modelo no soporta análisis de importancia de features")
            feature_importance = []
        
        return {
            'feature_importance': feature_importance,
            'top_features': [f[0] for f in feature_importance[:5]],
            'model_type': model_results['best_model_name']
        }
        
    except Exception as e:
        logger.error(f"Error evaluando importancia de features: {str(e)}")
        raise


def generate_ml_insights(classification_results: Dict, regression_results: Dict) -> Dict[str, Any]:
    """
    Genera insights aplicables basados en los resultados de ML.
    
    Args:
        classification_results: Resultados del modelo de clasificación
        regression_results: Resultados del modelo de regresión
        
    Returns:
        Dict con insights y recomendaciones
    """
    logger.info("Generando insights de Machine Learning")
    
    try:
        insights = {
            'classification_insights': {
                'model_performance': classification_results['metrics'],
                'best_model': classification_results['best_model_name'],
                'feature_importance': classification_results.get('feature_importance', [])
            },
            'regression_insights': {
                'model_performance': regression_results['metrics'],
                'best_model': regression_results['best_model_name'],
                'feature_importance': regression_results.get('feature_importance', [])
            }
        }
        
        # Insights estratégicos
        strategic_insights = []
        
        # Análisis de clasificación
        if classification_results['metrics']['accuracy'] > 0.7:
            strategic_insights.append(
                f"Modelo de clasificación efectivo (Accuracy: {classification_results['metrics']['accuracy']:.3f}). "
                "Puede predecir el rango del jugador basado en métricas de partida."
            )
        
        # Análisis de regresión
        if regression_results['metrics']['r2'] > 0.5:
            strategic_insights.append(
                f"Modelo de regresión útil (R²: {regression_results['metrics']['r2']:.3f}). "
                "Puede predecir el placement con precisión moderada."
            )
        
        # Variables críticas identificadas
        critical_vars = set()
        if classification_results.get('feature_importance'):
            critical_vars.update([f[0] for f in classification_results['feature_importance'][:3]])
        if regression_results.get('feature_importance'):
            critical_vars.update([f[0] for f in regression_results['feature_importance'][:3]])
        
        if critical_vars:
            strategic_insights.append(
                f"Variables críticas identificadas: {', '.join(list(critical_vars)[:5])}. "
                "Estas métricas son las más importantes para el rendimiento."
            )
        
        insights['strategic_insights'] = strategic_insights
        insights['critical_variables'] = list(critical_vars)
        
        logger.info(f"Insights generados: {len(strategic_insights)} recomendaciones estratégicas")
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generando insights: {str(e)}")
        raise


def save_ml_models(classification_results: Dict, regression_results: Dict, 
                  output_path: str) -> Dict[str, str]:
    """
    Guarda los modelos entrenados y sus métricas.
    
    Args:
        classification_results: Resultados del modelo de clasificación
        regression_results: Resultados del modelo de regresión
        output_path: Ruta base para guardar archivos
        
    Returns:
        Dict con rutas de archivos guardados
    """
    logger.info("Guardando modelos de Machine Learning")
    
    try:
        saved_files = {}
        
        # Guardar modelo de clasificación
        if classification_results:
            clf_path = f"{output_path}/classification_model.joblib"
            joblib.dump(classification_results['best_model'], clf_path)
            saved_files['classification_model'] = clf_path
            
            if classification_results.get('scaler'):
                scaler_path = f"{output_path}/classification_scaler.joblib"
                joblib.dump(classification_results['scaler'], scaler_path)
                saved_files['classification_scaler'] = scaler_path
        
        # Guardar modelo de regresión
        if regression_results:
            reg_path = f"{output_path}/regression_model.joblib"
            joblib.dump(regression_results['best_model'], reg_path)
            saved_files['regression_model'] = reg_path
            
            if regression_results.get('scaler'):
                scaler_path = f"{output_path}/regression_scaler.joblib"
                joblib.dump(regression_results['scaler'], scaler_path)
                saved_files['regression_scaler'] = scaler_path
        
        # Guardar métricas
        metrics = {
            'classification_metrics': classification_results.get('metrics', {}),
            'regression_metrics': regression_results.get('metrics', {}),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        metrics_path = f"{output_path}/ml_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        saved_files['metrics'] = metrics_path
        
        logger.info(f"Modelos guardados: {list(saved_files.keys())}")
        
        return saved_files
        
    except Exception as e:
        logger.error(f"Error guardando modelos: {str(e)}")
        raise
