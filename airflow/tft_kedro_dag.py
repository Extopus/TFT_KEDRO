"""
DAG de Airflow para orquestar pipelines de TFT Kedro.

Este DAG ejecuta los pipelines de clasificación y regresión en secuencia,
permitiendo la programación automática de las ejecuciones y el monitoreo
del estado de los pipelines.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

import sys
import os
import json
from pathlib import Path
import logging
import logging.config
from typing import Any

# Añadir el proyecto Kedro al PATH
project_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_path))

# Configurar logging desde el archivo JSON
logging_config_path = Path(__file__).parent / "logging.json"
if logging_config_path.exists():
    with open(logging_config_path) as f:
        logging_config = json.load(f)
    logging.config.dictConfig(logging_config)

# Importar funciones de Kedro
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Configuración por defecto para el DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['tu_email@ejemplo.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_kedro_pipeline(pipeline_name: str, **kwargs):
    """
    Ejecuta un pipeline específico de Kedro.
    
    Args:
        pipeline_name: Nombre del pipeline a ejecutar
    """
    logger = logging.getLogger("airflow.tft_kedro_dag")
    logger.info("Iniciando ejecución de pipeline Kedro: %s", pipeline_name)

    try:
        # Asegurar que el CWD esté en la raíz del proyecto para que Kedro encuentre conf
        os.chdir(str(project_path))

        # Inicializar proyecto Kedro (usar string para compatibilidad)
        bootstrap_project(str(project_path))

        # Crear sesión de Kedro y ejecutar el pipeline
        with KedroSession.create(project_path=str(project_path)) as session:
            session.run(pipeline_name=pipeline_name)

        logger.info("Pipeline %s ejecutado correctamente", pipeline_name)
    except Exception as exc:  # noqa: BLE001 - queremos capturar y loggear errores
        logger.exception("Error ejecutando pipeline %s: %s", pipeline_name, exc)
        # Re-raise para que Airflow marque la tarea como fallida
        raise

# Crear DAG
dag = DAG(
    'tft_kedro_pipelines',
    default_args=default_args,
    description='DAG para ejecutar pipelines de TFT Kedro',
    schedule_interval=timedelta(days=1),  # Ejecutar diariamente
    start_date=days_ago(1),
    tags=['kedro', 'machine_learning', 'tft'],
)

# Tarea para ejecutar pipeline de clasificación
run_classification = PythonOperator(
    task_id='run_classification_pipeline',
    python_callable=run_kedro_pipeline,
    op_kwargs={'pipeline_name': 'classification_rf'},
    dag=dag,
)

# Tarea para ejecutar pipeline de regresión
run_regression = PythonOperator(
    task_id='run_regression_pipeline',
    python_callable=run_kedro_pipeline,
    op_kwargs={'pipeline_name': 'regression_rf'},
    dag=dag,
)

# Definir el orden de ejecución
run_classification >> run_regression  # Ejecutar clasificación antes que regresión