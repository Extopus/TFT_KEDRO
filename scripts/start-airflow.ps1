# Script para construir y arrancar el stack de Airflow en Docker
param(
    [switch]$Clean,
    [switch]$Init
)

$ErrorActionPreference = "Stop"
$ComposeFile = "docker-compose.airflow.yml"

# Función para mostrar mensajes con formato
function Write-Step {
    param($Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

# Limpiar si se solicita
if ($Clean) {
    Write-Step "Limpiando contenedores y volúmenes anteriores"
    docker compose -f $ComposeFile down -v
}

# Construir imágenes
Write-Step "Construyendo imágenes de Docker"
docker compose -f $ComposeFile build

# Iniciar postgres
Write-Step "Iniciando postgres"
docker compose -f $ComposeFile up -d postgres

# Esperar a que postgres esté listo
Write-Step "Esperando a que postgres esté disponible..."
Start-Sleep -Seconds 10

if ($Init) {
    Write-Step "Inicializando base de datos de Airflow"
    docker compose -f $ComposeFile run --rm airflow-webserver airflow db init

    Write-Step "Creando usuario admin"
    docker compose -f $ComposeFile run --rm airflow-webserver `
        airflow users create `
            --username admin `
            --firstname Admin `
            --lastname User `
            --role Admin `
            --email admin@example.com `
            --password admin
}

# Iniciar servicios
Write-Step "Iniciando servicios de Airflow"
docker compose -f $ComposeFile up -d

Write-Step "Stack de Airflow iniciado"
Write-Host "UI disponible en: http://localhost:8080"
Write-Host "Usuario: admin"
Write-Host "Contraseña: admin"

# Mostrar logs
Write-Step "Mostrando logs (Ctrl+C para salir)"
docker compose -f $ComposeFile logs -f