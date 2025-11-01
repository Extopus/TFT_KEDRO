Resumen rápido para ejecutar Airflow con Docker (proyecto TFT_KEDRO)

Archivos creados:
- docker/airflow/Dockerfile  -> imagen basada en apache/airflow que instala tus dependencias (requirements.txt)
- docker-compose.airflow.yml -> stack: postgres, redis, airflow-webserver, airflow-scheduler

Pasos para levantar (PowerShell, desde la raíz del repo):

1) Construir la imagen:

```powershell
docker compose -f docker-compose.airflow.yml build
```

2) Levantar Postgres y Redis (opcionalmente en background):

```powershell
docker compose -f docker-compose.airflow.yml up -d postgres redis
```

3) Inicializar la base de datos de Airflow y crear el usuario admin:

```powershell
# Inicializar la DB de Airflow
docker compose -f docker-compose.airflow.yml run --rm airflow-webserver airflow db init

# Crear usuario admin (elige usuario/clave)
docker compose -f docker-compose.airflow.yml run --rm airflow-webserver \
  airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
```

4) Levantar webserver y scheduler:

```powershell
docker compose -f docker-compose.airflow.yml up -d
```

5) Abrir UI de Airflow en:

    http://localhost:8080  (usuario: admin / contraseña: admin)

Notas y recomendaciones:
- El DAG `airflow/tft_kedro_dag.py` ya está en tu repo y se monta en `/opt/airflow/project/airflow` dentro del contenedor — Airflow lo detectará como DAG.
- La imagen instala `requirements.txt` (tu entorno Kedro). Si hay paquetes nativos que requieran compilación en Linux (p. ej. algunas versiones de pyarrow, deltalake, etc.), el build puede fallar en Windows, en cuyo caso es preferible ejecutar Docker en WSL2 para construir la imagen o usar una CI que construya la imagen.
- Si tu proyecto depende de variables de entorno especiales o de credenciales, exporta esas variables en la sección `environment` del servicio `airflow-webserver` y `airflow-scheduler` en `docker-compose.airflow.yml`.
- Logs y plugins se persisten localmente en `./logs` y `./plugins`.

Si quieres, puedo:
- Ajustar la versión de Airflow en el Dockerfile si necesitas una versión concreta.
- Añadir un `Makefile` o scripts PowerShell para simplificar los comandos.
- Probar levantar el stack aquí (si tu entorno lo permite) y validar que el DAG aparece en la UI.
