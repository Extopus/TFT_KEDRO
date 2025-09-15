<<<<<<< HEAD
# TFT_KEDRO
=======
# TFT Kedro Project

Proyecto de análisis de datos de Teamfight Tactics (TFT) utilizando el framework Kedro para la gestión de pipelines de datos.

## Descripción

Este proyecto analiza datos de partidas de TFT de diferentes rangos competitivos (Challenger, Grandmaster, Platinum) para extraer insights y patrones que puedan ser útiles para el modelado predictivo.

## Estructura del Proyecto

```
TFT_KEDRO/
├── conf/                    # Configuraciones de Kedro
├── data/                    # Datos del proyecto
│   ├── 01_raw/             # Datos crudos
│   ├── 02_intermediate/    # Datos limpios
│   └── 04_feature/         # Datos con features
├── notebooks/               # Jupyter notebooks para análisis
├── src/tft_kedro/          # Código fuente del proyecto
└── pipelines/              # Pipelines de Kedro
```

## Instalación y Configuración

### 1. Crear entorno virtual
```bash
python -m venv .venv
```

### 2. Activar entorno virtual
```bash
# En Windows
.venv\Scripts\activate

# En Linux/Mac
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar Jupyter Notebooks
```bash
jupyter notebook --NotebookApp.token=''
```

## Ejecución de Pipelines

### Pipeline de Comprensión del Negocio
```bash
kedro run --pipeline business_understanding
```

### Pipeline de Limpieza de Datos
```bash
kedro run --pipeline data_cleaning
```

### Pipeline de Ingeniería de Features
```bash
kedro run --pipeline feature_engineering
```

### Visualización de Pipelines
```bash
kedro viz
```

## Pipelines Disponibles

1. **business_understanding**: Análisis exploratorio inicial de los datos
2. **data_cleaning**: Limpieza y preprocesamiento de datos
3. **feature_engineering**: Creación y transformación de variables

## Datos

El proyecto incluye datos de partidas de TFT de tres rangos competitivos:
- Challenger
- Grandmaster  
- Platinum

Los datos se procesan a través de tres etapas:
- **Raw**: Datos originales sin procesar
- **Intermediate**: Datos limpios y preprocesados
- **Feature**: Datos con variables de ingeniería aplicadas

## Requisitos

- Python 3.8+
- Kedro
- Pandas
- NumPy
- Jupyter Notebook

## Contribución

Para contribuir al proyecto, sigue estos pasos:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request
