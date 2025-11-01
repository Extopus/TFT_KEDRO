FROM apache/airflow:2.9.0-python3.9

# Mantener usuario airflow (NO cambiar a root)
USER airflow

# Directorio de trabajo y entorno
WORKDIR /opt/airflow

# Copiar requirements
COPY --chown=airflow:root requirements.txt .

# Instalar dependencias como usuario airflow
RUN pip install --no-cache-dir -r requirements.txt

# Asegurar que los paquetes instalados con --user estén en el PATH
ENV PATH=/opt/airflow/.local/bin:$PATH

# Crear directorios necesarios
USER root
RUN mkdir -p /opt/airflow/airflow/logs && \
    chown -R airflow:root /opt/airflow/airflow

# Volver a usuario airflow
USER airflow

# Ajustes por defecto
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/project/airflow \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    PYTHONPATH=/opt/airflow/project:$PYTHONPATH
