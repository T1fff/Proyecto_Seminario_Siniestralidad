# Modelación Predictiva de la Siniestralidad en Seguros de Automóviles

**Un Análisis Comparativo entre Modelos Lineales Generalizados (GLM), Algoritmos de
Machine Learning y Redes Neuronales**

Seminario investigativo de Ciencia de Datos aplicado al sector asegurador.

## Objetivo general

Comparar el desempeño de los modelos lineales generalizados (GLM) tradicionales y los
algoritmos de Machine Learning para la estimación de la frecuencia, severidad y prima
pura en seguros de automóviles, considerando su capacidad predictiva e interpretación.

## Objetivos específicos

1. Caracterizar las variables explicativas asociadas a la frecuencia, severidad y prima
   pura de los siniestros mediante un análisis exploratorio de datos.
2. Desarrollar modelos GLM utilizando Poisson, Gamma y Tweedie.
3. Desarrollar modelos XGBoost, LightGBM, CatBoost y redes neuronales.
4. Comparar ambos enfoques mediante métricas cuantitativas de capacidad predictiva y
   análisis cualitativo de interpretabilidad actuarial.

## Estado actual del proyecto

Esta entrega corresponde **únicamente** a la primera etapa de la investigación:

| Notebook | Contenido |
|---|---|
| `notebooks/01_Analisis_Exploratorio_Datos.ipynb` | Inspección estructural, estadística descriptiva, exploración de frecuencia y severidad, relación entre tablas y diagnóstico de consistencia. |
| `notebooks/02_Analisis_Univariado.ipynb` | Análisis individual de cada variable numérica y categórica, con interpretación actuarial. |

Las etapas siguientes (análisis bivariado/multivariado, preprocesamiento,
construcción de frecuencia/severidad/prima pura, GLM, XGBoost/LightGBM/CatBoost, redes
neuronales, evaluación, comparación e interpretabilidad) **no forman parte de esta
entrega** y se desarrollarán en fases posteriores del seminario.

## Datos

Se utiliza el conjunto **freMTPL2**, correspondiente a una cartera real y anonimizada de
pólizas de responsabilidad civil de automóviles (*Motor Third-Party Liability*) en
Francia. Ver `data/raw/README.md` para el diccionario de variables completo, la
procedencia de los datos y una nota importante sobre la naturaleza agregada del archivo
de severidad incluido en este proyecto.

| Archivo | Descripción |
|---|---|
| `data/raw/freMTPL2freq.csv` | ~678.000 pólizas, variables de riesgo y `ClaimNb`. |
| `data/raw/freMTPL2sev.csv` | Monto de siniestro agregado por póliza (`ClaimAmount`). |

## Estructura del repositorio

```text
Proyecto_Seminario_Siniestralidad/
│
├── data/
│   ├── raw/                  # Datos originales (freMTPL2freq.csv, freMTPL2sev.csv) + diccionario de variables
│   └── processed/            # Reservado para datasets procesados en etapas futuras
│
├── notebooks/
│   ├── 01_Analisis_Exploratorio_Datos.ipynb
│   └── 02_Analisis_Univariado.ipynb
│
├── src/
│   └── utils.py              # Funciones auxiliares reutilizables (carga de datos, resúmenes, gráficos)
│
├── docs/
│   └── index.md              # Portada del sitio de documentación (MyST)
│
├── .github/workflows/
│   └── deploy.yml            # Despliegue automático del sitio a GitHub Pages
│
├── myst.yml                  # Configuración del sitio (MyST / Jupyter Book)
├── requirements.txt
├── .gitignore
└── README.md
```

Esta organización sigue el mismo patrón profesional de un proyecto de referencia previo
del autor (EDA de transacciones de pago en la Unión Europea), adaptado a las necesidades
específicas de este proyecto actuarial: se separan explícitamente `data/raw` y
`data/processed`, se agrupan los notebooks en `notebooks/`, se centralizan las funciones
reutilizables en `src/`, y se mantiene `myst.yml` junto con un workflow de GitHub Actions
para el despliegue del sitio, de forma equivalente al mecanismo de despliegue del
proyecto de referencia.

## Cómo ejecutar el proyecto

1. Clonar o descomprimir el repositorio.
2. Colocar `freMTPL2freq.csv` y `freMTPL2sev.csv` en `data/raw/` (ya incluidos en esta
   entrega; ver `data/raw/README.md` si necesita obtenerlos nuevamente).
3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Abrir y ejecutar, en orden, `notebooks/01_Analisis_Exploratorio_Datos.ipynb` y
   `notebooks/02_Analisis_Univariado.ipynb`.

El proyecto es compatible con **VS Code**, **Jupyter Notebook/JupyterLab** y **Google
Colab** (en este último caso, basta con montar el repositorio —o subir la carpeta
`data/raw/`— respetando la misma jerarquía de carpetas; las rutas son relativas y no
requieren rutas absolutas de ninguna máquina en particular).

## Despliegue del sitio de documentación

El repositorio incluye `myst.yml` y un workflow de GitHub Actions
(`.github/workflows/deploy.yml`) que construyen y publican un sitio de documentación a
partir de los notebooks ejecutados, de forma análoga al proyecto de referencia. Para
publicarlo, basta con habilitar GitHub Pages (fuente: GitHub Actions) en la
configuración del repositorio tras el primer push a `main`.

## Alcance metodológico de esta entrega

Por diseño, esta entrega **no incluye**: tratamiento de valores faltantes, imputación,
eliminación de outliers, transformación definitiva de variables, feature engineering,
partición train/test, validación cruzada, balanceo de clases, optimización de
hiperparámetros, ni ningún modelo (GLM, Machine Learning o redes neuronales) ni sus
métricas de evaluación. Todos los hallazgos de calidad de datos se documentan de forma
descriptiva para ser abordados en la etapa de preprocesamiento correspondiente.
