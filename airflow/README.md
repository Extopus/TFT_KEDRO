# Orquestación con Apache Airflow

Este directorio contiene la configuración necesaria para orquestar los pipelines de Kedro usando Apache Airflow.

## Estructura
- `tft_kedro_dag.py`: Define el DAG que ejecuta los pipelines de clasificación y regresión
- `requirements.txt`: Dependencias necesarias para ejecutar el DAG

## Configuración

1. Instalar dependencias:
```bash
pip install -r airflow/requirements.txt
```

2. Configurar variable de entorno AIRFLOW_HOME:
```bash
export AIRFLOW_HOME=~/airflow
```

3. Inicializar la base de datos de Airflow:
```bash
airflow db init
```

4. Crear usuario admin:
```bash
airflow users create \
    --username admin \
    --firstname FIRST_NAME \
    --lastname LAST_NAME \
    --role Admin \
    --email your@email.com \
    --password admin
```

5. Copiar o enlazar el archivo DAG:
```bash
ln -s $(pwd)/airflow/tft_kedro_dag.py ~/airflow/dags/
```

6. Iniciar el servidor web de Airflow:
```bash
airflow webserver --port 8080
```

7. En otra terminal, iniciar el scheduler:
```bash
airflow scheduler
```

## Uso

1. Acceder a la interfaz web de Airflow en http://localhost:8080

2. El DAG "tft_kedro_pipelines" debería estar visible

3. El DAG ejecutará:
   - Primero el pipeline de clasificación
   - Luego el pipeline de regresión
   
4. La ejecución está programada para ocurrir:
   - Diariamente (configurable en tft_kedro_dag.py)
   - Se puede ejecutar manualmente desde la interfaz web

## Monitoreo

- La interfaz web de Airflow muestra:
  - Estado de las ejecuciones
  - Logs de cada tarea
  - Duración de las ejecuciones
  - Historial de ejecuciones

## Personalización

Puedes modificar en tft_kedro_dag.py:
- Frecuencia de ejecución (schedule_interval)
- Configuración de reintentos
- Notificaciones por email
- Dependencias entre tareas