"""
src/clasificacion.py
Clasificación de la severidad media por siniestro en tres bandas ordinales
(barato, intermedio, caro) sobre freMTPL2.

Sigue la misma lógica que src/modelado.py (sección de severidad):
- Datos      : freMTPL2freq.csv + freMTPL2sev.csv, unidos con preparar_datos().
- Objetivo   : severidad media por siniestro = ClaimAmount / ClaimNb (sólo ClaimNb > 0),
               la misma respuesta que el modelo de severidad del Notebook 04.
- Predictores: FEATURES de src/modelado.py. Se excluyen IDpol (identificador),
               ClaimNb (es el denominador del objetivo y el peso de severidad) y
               Exposure (no se usa en severidad) para evitar fuga.
- Partición  : split a nivel de póliza estratificado por deciles/percentiles de la
               severidad (no sólo por ClaimNb>0, que aquí es constante).
- Cortes     : se estiman dentro de cada fold de validación y con todo el train para
               el modelo final. El test no interviene en la selección.
- Modelos    : GLM (regresión logística multinomial con penalización L2), XGBoost, LightGBM.
- Balanceo   : sin_balanceo | weights | smote | adasyn (sólo sobre el fold de entrenamiento).
- Checkpoints: CSV (una fila por combinación) + JSON (mejor por método y modelo) + joblib.
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, mean_absolute_error)
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.modelado import CAT_COLS, FEATURES, RANDOM_STATE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# 1. Configuración
# ──────────────────────────────────────────────────────────────────────
K = 3
ETIQUETAS = ["barato", "intermedio", "caro"]
METODOS_PRINCIPALES = ["jenks", "kmeans", "regla_cola"]
METODOS_REFERENCIA = ["ancho_igual_log", "fijos"]
METODOS = METODOS_PRINCIPALES + METODOS_REFERENCIA
MODELOS = ["GLM", "XGBoost", "LightGBM"]
BALANCEOS = ["sin_balanceo", "weights", "smote", "adasyn"]
NUM_FEATURES = [c for c in FEATURES if c not in CAT_COLS]          # numéricas (incl. LogDensity)
EXCLUIDAS = ["IDpol", "ClaimNb", "Exposure", "ClaimAmount"]        # fuga / identificador / objetivo
CUANTILES_ESTRATO = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)             # cortes para estratificar la partición

ESPACIOS = {
    "GLM": {"C": [0.01, 0.1, 1.0, 10.0]},                          # C = 1/alpha (penalización L2)
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
}


# ──────────────────────────────────────────────────────────────────────
# 2. Datos y partición
# ──────────────────────────────────────────────────────────────────────
def preparar_severidad(df: pd.DataFrame):
    """
    A partir de df = preparar_datos(freq, sev) devuelve (X, sev) sólo con pólizas
    con siniestro. X contiene FEATURES; sev = ClaimAmount / ClaimNb.
    """
    d = df[df["ClaimNb"] > 0].reset_index(drop=True)
    X = d[FEATURES].copy()
    sev = (d["ClaimAmount"] / d["ClaimNb"]).to_numpy(dtype=float)
    assert not X.isna().any().any(), "Hay NaN en los predictores."
    assert (sev > 0).all(), "Hay severidades no positivas."
    return X, sev


def estrato_severidad(sev, cuantiles=CUANTILES_ESTRATO):
    """
    Estrato = tramo de log1p(severidad) entre cuantiles. Sólo sirve para repartir
    proporcionalmente las colas entre train y test y para armar los folds; no entra
    como predictor.
    """
    l = np.log1p(np.asarray(sev, dtype=float))
    bordes = np.unique(np.quantile(l, cuantiles))
    return np.digitize(l, bordes)


def dividir_estratificado(sev, test_size=0.20, seed=RANDOM_STATE):
    """Devuelve índices de train y test, estratificados por estrato_severidad."""
    estrato = estrato_severidad(sev)
    idx = np.arange(len(sev))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=seed,
                                      stratify=estrato)
    return idx_tr, idx_te, estrato


def resumen_particion(sev_tr, sev_te, cortes_tr_por_metodo):
    """
    Conteo por banda en train y test para cada método. Sirve para detectar casos
    como 2 observaciones en train y 20 en test.
    """
    log_tr, log_te = np.log1p(sev_tr), np.log1p(sev_te)
    filas = []
    for metodo, cortes in cortes_tr_por_metodo.items():
        y_tr = etiquetar(log_tr, cortes)
        y_te = etiquetar(log_te, cortes)
        for k, e in enumerate(ETIQUETAS):
            filas.append({"metodo": metodo, "banda": e,
                          "n_train": int((y_tr == k).sum()),
                          "n_test": int((y_te == k).sum())})
    return pd.DataFrame(filas)


# ──────────────────────────────────────────────────────────────────────
# 3. Métodos para definir las bandas (todos reciben log1p(severidad))
# ──────────────────────────────────────────────────────────────────────
def cortes_jenks(valores, k=K, muestra=20000, seed=RANDOM_STATE):
    """Jenks natural breaks: minimiza la varianza intra-clase (submuestra si es grande)."""
    import jenkspy
    rng = np.random.default_rng(seed)
    sub = rng.choice(valores, size=min(muestra, len(valores)), replace=False)
    brks = jenkspy.jenks_breaks(sub.tolist(), n_classes=k)   # k+1 valores con mín y máx
    return np.asarray(brks[1:-1], dtype=float)


def cortes_kmeans(valores, k=K, muestra=20000, seed=RANDOM_STATE):
    """K-means 1D. Cortes = puntos medios entre centroides consecutivos."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(valores, size=min(muestra, len(valores)), replace=False).reshape(-1, 1)
    centros = np.sort(KMeans(n_clusters=k, n_init=10, random_state=seed)
                      .fit(sub).cluster_centers_.ravel())
    return (centros[:-1] + centros[1:]) / 2


def cortes_regla_cola(valores, k=K):
    """
    Regla de cola de Wang et al. (arXiv:1911.02418), método M2 (raíz cuadrada).
    Cola superior: umbral b = observación de rango n − m, con m = round(√n).
    Cola inferior: EXTENSIÓN PROPIA; el paper sólo define el umbral superior.
    Sólo sirve para k = 3 (dos cortes).
    """
    if k != 3:
        raise ValueError("cortes_regla_cola está definida para k = 3.")
    s = np.sort(valores)
    n = len(s)
    m = int(round(np.sqrt(n)))
    b_sup = s[n - m - 1]
    b_inf = s[m]
    return np.array([b_inf, b_sup])


def cortes_ancho_igual_log(valores, k=K):
    """Referencia: k − 1 cortes equidistantes entre mín y máx en log1p."""
    lo, hi = valores.min(), valores.max()
    return np.array([lo + (hi - lo) * j / k for j in range(1, k)])


def cortes_fijos(valores, k=K, cortes_monto=(1000.0, 3000.0)):
    """Referencia: cortes fijados a priori en unidades de monto (no de log)."""
    c = np.sort(np.log1p(np.asarray(cortes_monto, dtype=float)))
    if len(c) != k - 1:
        raise ValueError(f"cortes_monto debe tener {k - 1} valores para k = {k}.")
    return c


def obtener_funciones_cortes(cortes_monto=(1000.0, 3000.0), muestra=20000, seed=RANDOM_STATE):
    return {
        "jenks": partial(cortes_jenks, muestra=muestra, seed=seed),
        "kmeans": partial(cortes_kmeans, muestra=muestra, seed=seed),
        "regla_cola": cortes_regla_cola,
        "ancho_igual_log": cortes_ancho_igual_log,
        "fijos": partial(cortes_fijos, cortes_monto=cortes_monto),
    }


def etiquetar(valores_log, cortes):
    """0 = barato, 1 = intermedio, 2 = caro."""
    return np.digitize(valores_log, cortes)


def gvf(valores_log, cortes):
    """Goodness of variance fit: 1 − SDCM/SDAM. Cerca de 1 = bandas homogéneas."""
    y = etiquetar(valores_log, cortes)
    sdam = ((valores_log - valores_log.mean()) ** 2).sum()
    sdcm = sum(((valores_log[y == i] - valores_log[y == i].mean()) ** 2).sum()
               for i in np.unique(y))
    return float(1.0 - sdcm / sdam)


# ──────────────────────────────────────────────────────────────────────
# 4. Preprocesamiento, modelos y balanceo
# ──────────────────────────────────────────────────────────────────────
def crear_preprocesador():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ], remainder="drop")


def crear_clasificador(nombre, params):
    if nombre == "GLM":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **params)
    if nombre == "XGBoost":
        return xgb.XGBClassifier(objective="multi:softprob", tree_method="hist",
                                 n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
                                 **params)
    if nombre == "LightGBM":
        return lgb.LGBMClassifier(objective="multiclass", random_state=RANDOM_STATE,
                                  n_jobs=-1, verbose=-1, subsample_freq=1, **params)
    raise ValueError(f"Modelo desconocido: {nombre}")


def _estrategia_sobremuestreo(y, ratio):
    """
    Lleva cada banda minoritaria a ratio × (tamaño de la mayoritaria).
    Equivalente al sampling_ratio = 0.2 de SMOTE/ADASYN del Notebook 04, extendido a 3 clases.
    """
    conteo = np.bincount(y, minlength=K)
    objetivo = int(ratio * conteo.max())
    return {c: objetivo for c in range(K) if 0 < conteo[c] < objetivo}


def construir_pipeline(modelo, params, balanceo, y, ratio=0.2):
    pasos = [("prep", crear_preprocesador())]
    if balanceo in ("smote", "adasyn"):
        estrat = _estrategia_sobremuestreo(np.asarray(y), ratio)
        if balanceo == "smote":
            pasos.append(("samp", SMOTE(sampling_strategy=estrat, random_state=RANDOM_STATE)))
        else:
            pasos.append(("samp", ADASYN(sampling_strategy=estrat, random_state=RANDOM_STATE)))
    elif balanceo not in ("sin_balanceo", "weights"):
        raise ValueError(f"balanceo desconocido: {balanceo}")
    pasos.append(("clf", crear_clasificador(modelo, params)))
    return ImbPipeline(pasos)


def ajustar(modelo, params, balanceo, X, y, ratio=0.2):
    pipe = construir_pipeline(modelo, params, balanceo, y, ratio)
    kwargs = {}
    if balanceo == "weights":
        kwargs["clf__sample_weight"] = compute_sample_weight("balanced", y)
    return pipe.fit(X, y, **kwargs)


def metricas(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=range(K), zero_division=0),
        "kappa_cuadratico": cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                              labels=list(range(K))),
        "mae_ordinal": mean_absolute_error(y_true, y_pred),
    }


# ──────────────────────────────────────────────────────────────────────
# 5. Checkpoints y experimentos
# ──────────────────────────────────────────────────────────────────────
CSV_COLS = ["fecha", "metodo", "modelo", "balanceo", "config_id", "hiperparametros",
            "cortes_monto", "gvf_train", "cv_f1_macro_media", "cv_f1_macro_std",
            "cv_kappa_media", "test_accuracy", "test_balanced_accuracy", "test_f1_macro",
            "test_kappa_cuadratico", "test_mae_ordinal", "tiempo_s", "status", "error"]


def leer_checkpoints(carpeta):
    ruta = Path(carpeta) / "checkpoints.csv"
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=CSV_COLS)


def mejores_por_metodo_modelo(ck):
    """Mejor balanceo y configuración por (método, modelo), según F1 macro de CV."""
    ok = ck[ck["status"] == "ok"]
    if ok.empty:
        return ok
    return ok.loc[ok.groupby(["metodo", "modelo"])["cv_f1_macro_media"].idxmax()]\
             .reset_index(drop=True)


def cargar_modelo(metodo, modelo, carpeta):
    return joblib.load(Path(carpeta) / "modelos" / f"{metodo}__{modelo}.joblib")


def _evaluar_combinacion(f, metodo, modelo, bal, cid, params, folds,
                         X_tr, log_tr, X_te, log_te, ratio):
    """Entrena una combinación con validación cruzada y con el train completo."""
    t0 = time.perf_counter()
    cortes_full = f(log_tr)
    y_tr = etiquetar(log_tr, cortes_full)
    y_te = etiquetar(log_te, cortes_full)

    f1s, kappas = [], []
    for itr, iva in folds:
        cortes_f = f(log_tr[itr])                              # cortes sólo con el fold de train
        pipe = ajustar(modelo, params, bal, X_tr.iloc[itr],
                       etiquetar(log_tr[itr], cortes_f), ratio)
        m_va = metricas(etiquetar(log_tr[iva], cortes_f), pipe.predict(X_tr.iloc[iva]))
        f1s.append(m_va["f1_macro"])
        kappas.append(m_va["kappa_cuadratico"])

    final = ajustar(modelo, params, bal, X_tr, y_tr, ratio)
    m_te = metricas(y_te, final.predict(X_te))
    fila = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "metodo": metodo, "modelo": modelo, "balanceo": bal, "config_id": cid,
        "hiperparametros": json.dumps(params, default=str),
        "cortes_monto": json.dumps([round(float(v), 4) for v in np.expm1(cortes_full)]),
        "gvf_train": gvf(log_tr, cortes_full),
        "cv_f1_macro_media": float(np.mean(f1s)), "cv_f1_macro_std": float(np.std(f1s)),
        "cv_kappa_media": float(np.mean(kappas)),
        **{f"test_{q}": float(v) for q, v in m_te.items()},
        "tiempo_s": round(time.perf_counter() - t0, 1), "status": "ok", "error": "",
    }
    return fila, final, cortes_full


def ejecutar_experimentos(X_tr, sev_tr, estrato_tr, X_te, sev_te, funciones,
                          metodos=METODOS, modelos=MODELOS, balanceos=BALANCEOS,
                          n_configs=3, cv_folds=3, carpeta="models_clasificacion",
                          ratio=0.2, reanudar=True, seed=RANDOM_STATE):
    """
    Recorre método x modelo x balanceo x configuración. Por cada combinación:
      - en cada fold, los cortes se estiman sólo con el entrenamiento de ese fold;
      - los cortes finales se estiman con todo el train y etiquetan el test;
      - se añade una fila a checkpoints.csv (escritura incremental);
      - si mejora el F1 de CV de su (método, modelo), se guarda el modelo y el JSON.
    Con reanudar=True se omiten las combinaciones ya registradas con status 'ok'.
    Los folds usan estrato_tr, que es fijo entre métodos, para que todos se comparen
    sobre las mismas particiones.
    """
    carpeta = Path(carpeta)
    (carpeta / "modelos").mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta / "checkpoints.csv"
    ruta_json = carpeta / "mejores_modelos.json"

    ck = leer_checkpoints(carpeta)
    hechos = set()
    if reanudar and not ck.empty:
        hechos = {(r.metodo, r.modelo, r.balanceo, int(r.config_id))
                  for r in ck[ck["status"] == "ok"].itertuples()}
    mejores = json.loads(ruta_json.read_text(encoding="utf-8")) if ruta_json.exists() else {}

    log_tr, log_te = np.log1p(np.asarray(sev_tr, float)), np.log1p(np.asarray(sev_te, float))
    folds = list(StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                 random_state=seed).split(X_tr, estrato_tr))

    combinaciones = []
    for metodo in metodos:
        for modelo in modelos:
            muestras = ParameterSampler(ESPACIOS[modelo], n_iter=n_configs, random_state=seed)
            for cid, params in enumerate(muestras):
                for bal in balanceos:
                    combinaciones.append((metodo, modelo, bal, cid, params))

    print(f"Combinaciones totales: {len(combinaciones)}")
    for k, (metodo, modelo, bal, cid, params) in enumerate(combinaciones, 1):
        clave = (metodo, modelo, bal, cid)
        if clave in hechos:
            continue
        try:
            fila, final, cortes_full = _evaluar_combinacion(
                funciones[metodo], metodo, modelo, bal, cid, params, folds,
                X_tr, log_tr, X_te, log_te, ratio)
        except Exception as e:                       # una combinación fallida no detiene la corrida
            fila = {c: np.nan for c in CSV_COLS}
            fila.update(fecha=datetime.now().isoformat(timespec="seconds"), metodo=metodo,
                        modelo=modelo, balanceo=bal, config_id=cid, status="error",
                        error=repr(e)[:300])
            pd.DataFrame([fila])[CSV_COLS].to_csv(ruta_csv, mode="a",
                                                  header=not ruta_csv.exists(), index=False)
            print(f"[{k}/{len(combinaciones)}] {metodo}|{modelo}|{bal}|cfg{cid} ERROR: {e!r}")
            continue

        pd.DataFrame([fila])[CSV_COLS].to_csv(ruta_csv, mode="a",
                                              header=not ruta_csv.exists(), index=False)

        grupo = f"{metodo}|{modelo}"
        m_te = {q: fila[f"test_{q}"] for q in ["accuracy", "balanced_accuracy", "f1_macro",
                                                 "kappa_cuadratico", "mae_ordinal"]}
        previo = mejores.get(grupo)
        if previo is None or fila["cv_f1_macro_media"] > previo["metricas_cv"]["f1_macro_media"]:
            ruta_modelo = carpeta / "modelos" / f"{metodo}__{modelo}.joblib"
            joblib.dump(final, ruta_modelo, compress=3)
            mejores[grupo] = {
                "metodo": metodo, "modelo": modelo, "balanceo": bal, "config_id": cid,
                "hiperparametros": params,
                "cortes_monto": [float(v) for v in np.expm1(cortes_full)],
                "etiquetas": ETIQUETAS, "gvf_train": fila["gvf_train"],
                "metricas_cv": {"f1_macro_media": fila["cv_f1_macro_media"],
                                "f1_macro_std": fila["cv_f1_macro_std"],
                                "kappa_media": fila["cv_kappa_media"]},
                "metricas_test": {q: float(v) for q, v in m_te.items()},
                "ruta_modelo": str(ruta_modelo), "fecha": fila["fecha"],
            }
            ruta_json.write_text(json.dumps(mejores, indent=2, ensure_ascii=False,
                                            default=str), encoding="utf-8")

        print(f"[{k}/{len(combinaciones)}] {grupo}|{bal}|cfg{cid} | CV F1={fila['cv_f1_macro_media']:.4f}"
              f" | test F1={fila['test_f1_macro']:.4f} | GVF={fila['gvf_train']:.4f} | {fila['tiempo_s']}s")

    return leer_checkpoints(carpeta)


# ──────────────────────────────────────────────────────────────────────
# 6. Comparación
# ──────────────────────────────────────────────────────────────────────
def bootstrap_f1(y_true, y_pred, n_boot=500, seed=RANDOM_STATE):
    """IC 95 % del F1 macro en test por bootstrap de pólizas."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = f1_score(y_true[i], y_pred[i], average="macro", labels=range(K), zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
