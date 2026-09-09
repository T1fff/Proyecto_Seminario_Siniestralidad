"""
utils.py
--------
Funciones auxiliares reutilizables para los notebooks de análisis exploratorio
y univariado del proyecto "Modelación Predictiva de la Siniestralidad en
Seguros de Automóviles".

Estas funciones se limitan a tareas de carga, resumen descriptivo y
visualización. No realizan imputación, eliminación de registros ni ninguna
transformación definitiva de los datos: esas etapas quedan fuera del alcance
de esta fase exploratoria del proyecto.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuración visual consistente para todos los notebooks
# ---------------------------------------------------------------------------
PALETTE = "viridis"
FIGSIZE_SINGLE = (8, 4.5)
FIGSIZE_WIDE = (11, 4.5)
FIGSIZE_GRID = (12, 8)


def set_plot_style() -> None:
    """Configura un estilo visual consistente para todos los gráficos."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams["figure.figsize"] = FIGSIZE_SINGLE
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["figure.dpi"] = 100


# ---------------------------------------------------------------------------
# Rutas y carga de datos
# ---------------------------------------------------------------------------
def get_data_path(filename: str) -> Path:
    """
    Devuelve una ruta robusta hacia un archivo en data/raw/, funcionando
    tanto si el notebook se ejecuta desde notebooks/ (ubicación por defecto)
    como si el proyecto se abre desde la raíz del repositorio.

    Parameters
    ----------
    filename : str
        Nombre del archivo dentro de data/raw/.

    Returns
    -------
    Path
        Ruta resuelta al archivo solicitado.

    Raises
    ------
    FileNotFoundError
        Si el archivo no se encuentra en ninguna de las rutas candidatas.
    """
    candidatos = [
        Path("../data/raw") / filename,   # ejecución típica desde notebooks/
        Path("data/raw") / filename,      # ejecución desde la raíz del repo
        Path("./") / filename,            # el archivo junto al notebook
    ]
    for ruta in candidatos:
        if ruta.exists():
            return ruta

    raise FileNotFoundError(
        f"No se encontró '{filename}' en ninguna de las rutas esperadas "
        f"({[str(r) for r in candidatos]}). "
        "Verifique que los datos estén ubicados en 'data/raw/' "
        "siguiendo las instrucciones del README."
    )


def load_freq(filename: str = "freMTPL2freq.csv") -> pd.DataFrame:
    """Carga la tabla de frecuencia (freMTPL2freq) desde data/raw/."""
    return pd.read_csv(get_data_path(filename))


def load_sev(filename: str = "freMTPL2sev.csv") -> pd.DataFrame:
    """Carga la tabla de severidad (freMTPL2sev) desde data/raw/."""
    return pd.read_csv(get_data_path(filename))


# ---------------------------------------------------------------------------
# Resúmenes descriptivos
# ---------------------------------------------------------------------------
def resumen_numerico(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Genera un resumen descriptivo extendido (incluye asimetría) para un
    conjunto de variables numéricas.
    """
    resumen = df[columnas].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    resumen["skew"] = df[columnas].skew()
    resumen["missing"] = df[columnas].isna().sum()
    resumen["missing_%"] = (df[columnas].isna().mean() * 100).round(3)
    return resumen.round(3)


def tabla_frecuencias(df: pd.DataFrame, columna: str, top_n: int = None) -> pd.DataFrame:
    """
    Construye una tabla de frecuencias absolutas y relativas para una
    variable categórica, opcionalmente limitada a las top_n categorías
    más frecuentes.
    """
    conteo = df[columna].value_counts(dropna=False)
    proporcion = df[columna].value_counts(normalize=True, dropna=False) * 100
    tabla = pd.DataFrame(
        {"frecuencia_absoluta": conteo, "frecuencia_relativa_%": proporcion.round(2)}
    )
    if top_n is not None:
        tabla = tabla.head(top_n)
    return tabla


def resumen_categorico(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Resume la cardinalidad y concentración de un conjunto de variables
    categóricas: número de categorías, categoría dominante y su peso relativo.
    """
    filas = []
    for col in columnas:
        conteo = df[col].value_counts(normalize=True)
        filas.append(
            {
                "variable": col,
                "n_categorias": df[col].nunique(),
                "categoria_dominante": conteo.index[0],
                "peso_categoria_dominante_%": round(conteo.iloc[0] * 100, 2),
                "categoria_menos_frecuente": conteo.index[-1],
                "peso_categoria_menos_frecuente_%": round(conteo.iloc[-1] * 100, 4),
            }
        )
    return pd.DataFrame(filas).set_index("variable")


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def hist_box(
    serie: pd.Series,
    titulo: str,
    xlabel: str,
    bins: int = 50,
    log_x: bool = False,
    color: str = None,
):
    """
    Dibuja un histograma con KDE y un boxplot horizontal alineado debajo,
    para caracterizar tendencia central, dispersión y forma de la
    distribución de una variable numérica.
    """
    fig, (ax_hist, ax_box) = plt.subplots(
        nrows=2, ncols=1, figsize=FIGSIZE_SINGLE,
        gridspec_kw={"height_ratios": [4, 1]}, sharex=True,
    )
    sns.histplot(serie, bins=bins, kde=True, ax=ax_hist, color=color)
    ax_hist.set_title(titulo)
    ax_hist.set_ylabel("Frecuencia")

    sns.boxplot(x=serie, ax=ax_box, color=color)
    ax_box.set_xlabel(xlabel)

    if log_x:
        ax_hist.set_xscale("log")
        ax_box.set_xscale("log")

    plt.tight_layout()
    plt.show()


def barplot_top_categorias(
    df: pd.DataFrame, columna: str, top_n: int, titulo: str, xlabel: str
):
    """Grafica las top_n categorías más frecuentes de una variable categórica."""
    conteo = df[columna].value_counts().head(top_n)
    etiquetas = conteo.index.astype(str)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.barplot(
        x=etiquetas, y=conteo.values, hue=etiquetas, ax=ax,
        palette=PALETTE, legend=False,
    )
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Número de pólizas")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


def ecdf(serie: pd.Series):
    """Calcula la función de distribución empírica (ECDF) de una variable."""
    x = np.sort(serie.dropna().values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y
