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
