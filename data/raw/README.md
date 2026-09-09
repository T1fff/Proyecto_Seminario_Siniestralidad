# Datos crudos — freMTPL2

## Procedencia

El conjunto **freMTPL2** corresponde a una cartera de pólizas de responsabilidad civil
de automóviles (*Motor Third-Party Liability*) en Francia, publicada originalmente por
Arthur Charpentier como material complementario de *Computational Actuarial Science
with R* (CRC Press, 2018) y distribuida públicamente a través de:

- OpenML: `freMTPL2freq` (`data_id=41214`) y `freMTPL2sev` (`data_id=41215`)
- Kaggle: `karansarpal/freMTPL2-french-motor-tpl-insurance-claims`
- Paquete R `CASdatasets` (`https://github.com/dutangc/CASdatasets`)

## Archivos de este proyecto

| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| `freMTPL2freq.csv` | 678.007 | 12 | Variables de riesgo y número de siniestros (`ClaimNb`) por póliza. |
| `freMTPL2sev.csv` | 24.938 | 2 | Monto de siniestro (`ClaimAmount`) **agregado por póliza** (ver nota abajo). |

> **Nota importante sobre `freMTPL2sev.csv`.** La tabla de severidad oficial de
> `CASdatasets` registra **cada siniestro de forma individual** (una póliza con varios
> siniestros puede tener varias filas, y el conteo total de registros ronda 26.639).
> El archivo `freMTPL2sev.csv` incluido en este proyecto contiene el **monto total de
> siniestros agregado por póliza** (`IDpol`, `ClaimAmount`), obtenido a partir de la
> misma fuente pública verificada. Esta diferencia se documenta y se tiene en cuenta
> explícitamente en la Sección 11 del Notebook 01 (relación estructural entre tablas).
> Ningún valor fue simulado o inventado: todos los montos provienen de la fuente
> original.
>
> Si para una etapa posterior de la investigación se requiere la tabla de severidad
> **no agregada** (con un registro por siniestro individual), esta puede obtenerse
> directamente desde OpenML (`data_id=41215`) o desde el dataset de Kaggle indicado
> arriba, y colocarse en esta misma carpeta bajo el nombre `freMTPL2sev.csv`.

## Diccionario de variables — `freMTPL2freq.csv`

| Variable | Tipo | Descripción |
|---|---|---|
| `IDpol` | entero | Identificador único de póliza (llave de relación con `freMTPL2sev.csv`). |
| `Exposure` | numérica continua | Duración de la observación de la póliza, en fracción de año (0, 1]. |
| `Area` | categórica | Código de área (6 categorías: A–F). |
| `VehPower` | numérica discreta/ordinal | Potencia del vehículo. |
| `VehAge` | numérica discreta | Antigüedad del vehículo, en años. |
| `DrivAge` | numérica discreta | Edad del conductor, en años. |
| `BonusMalus` | numérica discreta | Coeficiente bonus-malus (100 = neutro; sistema francés). |
| `VehBrand` | categórica | Marca del vehículo (categorías anonimizadas B1–B14). |
| `VehGas` | categórica | Tipo de combustible (`Diesel` / `Regular`). |
| `Density` | numérica continua | Densidad poblacional (hab/km²) de la zona de residencia. |
| `Region` | categórica | Región francesa de residencia del conductor. |
| `ClaimNb` | numérica discreta (objetivo frecuencia) | Número de siniestros reportados durante `Exposure`. |

## Diccionario de variables — `freMTPL2sev.csv`

| Variable | Tipo | Descripción |
|---|---|---|
| `IDpol` | entero | Identificador de póliza (llave de relación con `freMTPL2freq.csv`). |
| `ClaimAmount` | numérica continua (objetivo severidad) | Monto total de siniestro(s) de la póliza. |

## Ubicación esperada

Ambos archivos deben residir en `data/raw/` para que los notebooks los encuentren
mediante rutas relativas robustas (ver `src/utils.py::get_data_path`). No es necesario
modificar ninguna ruta si se respeta esta estructura, tanto en ejecución local como en
Google Colab (montando el repositorio en la misma jerarquía de carpetas).
