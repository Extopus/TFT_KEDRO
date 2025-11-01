"""
Script para probar la función run_kedro_pipeline del DAG sin depender de Airflow.
"""
import sys
from pathlib import Path
import logging

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Importar la función desde el DAG
sys.path.append(str(Path(__file__).parent.parent))
from airflow.tft_kedro_dag import run_kedro_pipeline

def main():
    """Probar ejecución de ambos pipelines en secuencia."""
    try:
        # Simular la ejecución del DAG
        print("\n=== Probando pipeline de clasificación ===")
        run_kedro_pipeline("classification_rf")
        
        print("\n=== Probando pipeline de regresión ===")
        run_kedro_pipeline("regression_rf")
        
    except Exception as e:
        print(f"Error durante la prueba: {e}")
        raise

if __name__ == "__main__":
    main()