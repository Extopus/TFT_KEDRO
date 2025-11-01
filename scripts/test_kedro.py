"""
Script para probar la ejecución de pipelines Kedro sin dependencias de Airflow.
"""
import sys
from pathlib import Path
import logging
import os

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kedro_test.log')
    ]
)

logger = logging.getLogger("kedro_test")

# Importar solo lo necesario de Kedro
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

def run_pipeline(pipeline_name: str):
    """
    Ejecuta un pipeline específico de Kedro.
    
    Args:
        pipeline_name: Nombre del pipeline a ejecutar
    """
    logger.info("Iniciando ejecución de pipeline Kedro: %s", pipeline_name)
    
    try:
        # Obtener ruta del proyecto (un nivel arriba de scripts/)
        project_path = Path(__file__).parent.parent.resolve()
        
        # Asegurar que estamos en el directorio del proyecto
        os.chdir(str(project_path))
        
        # Inicializar proyecto Kedro
        bootstrap_project(str(project_path))
        
        # Crear sesión de Kedro y ejecutar pipeline
        with KedroSession.create(project_path=str(project_path)) as session:
            session.run(pipeline_name=pipeline_name)
            
        logger.info("Pipeline %s ejecutado correctamente", pipeline_name)
        
    except Exception as exc:
        logger.exception("Error ejecutando pipeline %s: %s", pipeline_name, exc)
        raise

def main():
    """Probar ejecución de ambos pipelines en secuencia."""
    try:
        print("\n=== Probando pipeline de clasificación ===")
        run_pipeline("classification_rf")
        
        print("\n=== Probando pipeline de regresión ===")
        run_pipeline("regression_rf")
        
    except Exception as e:
        print(f"Error durante la prueba: {e}")
        raise

if __name__ == "__main__":
    main()