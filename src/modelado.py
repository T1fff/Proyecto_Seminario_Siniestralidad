"""
src/modelado.py
Pipeline de modelos baseline (GLM y árboles de gradient boosting) para
frecuencia y severidad sobre freMTPL2.

- Frecuencia : pérdida = Poisson deviance. Respuesta ClaimNb con Exposure.
- Severidad  : pérdida = Gamma deviance.   Respuesta ClaimAmount/ClaimNb,
               ponderada por ClaimNb (solo pólizas con siniestro).
- Búsqueda   : RandomizedSearchCV (CV con folds estratificados por ClaimNb>0).
- Balanceo   : sin_balanceo | weights | smote | adasyn (solo frecuencia).
- Checkpoints: CSV (una fila por combinación) + JSON (mejores hiperparámetros
               y métricas) + .joblib (modelo ajustado).
"""
from __future__ import annotations

import json
import os
import platform
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import ADASYN, SMOTE
from joblib import parallel_config
from scipy import sparse
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_tweedie_deviance,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# 1. Configuración global
# ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

NUM_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CAT_COLS = ["Area", "VehBrand", "VehGas", "Region"]
BIN_COLS = ["DrivAge", "VehAge", "BonusMalus"]       # se discretizan en el GLM
FEATURES = NUM_COLS + CAT_COLS + ["LogDensity"]

# columna de peso por tarea y si la respuesta es un conteo (se modela y/peso)
TASKS = {
    "freq": dict(weight_col="Exposure", target_is_count=True),
    "sev": dict(weight_col="ClaimNb", target_is_count=False),
}

CSV_COLS = [
    "timestamp", "task", "modelo", "balanceo", "n_iter", "cv_folds",
    "n_train", "n_test",
    "cv_deviance", "cv_deviance_std", "cv_mae", "cv_rmse", "cv_gini", "cv_bias",
    "test_deviance", "test_mae", "test_rmse", "test_gini", "test_bias",
    "train_deviance",
    "tiempo_busqueda_s", "tiempo_refit_s", "fit_time_medio_cv_s",
    "best_params", "modelo_path", "status", "error",
]


# ──────────────────────────────────────────────────────────────────────
# 2. Datos
# ──────────────────────────────────────────────────────────────────────
def preparar_datos(df_freq: pd.DataFrame, df_sev: pd.DataFrame) -> pd.DataFrame:
    """Une frecuencia y severidad por IDpol y crea LogDensity."""
    df = df_freq.merge(df_sev[["IDpol", "ClaimAmount"]], on="IDpol",
                       how="left", validate="one_to_one")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)
    df["LogDensity"] = np.log(df["Density"].clip(lower=1))
    return df


def dividir_datos(df: pd.DataFrame, test_size: float = 0.20):
    """Split a nivel de póliza, estratificado por ClaimNb>0."""
    estrato = (df["ClaimNb"] > 0).astype(int)
    idx_tr, idx_te = train_test_split(
        df.index, test_size=test_size, random_state=RANDOM_STATE, stratify=estrato
    )
    return (df.loc[idx_tr].reset_index(drop=True),
            df.loc[idx_te].reset_index(drop=True))


def armar_xy(df: pd.DataFrame, task: str):
    """
    freq: X = features + Exposure ; y = ClaimNb
    sev : X = features + ClaimNb   ; y = ClaimAmount / ClaimNb (solo ClaimNb>0)
    La columna de peso viaja dentro de X y el modelo la separa internamente.
    """
    if task == "freq":
        X = df[FEATURES + ["Exposure"]]
        y = df["ClaimNb"].astype(float)
    elif task == "sev":
        d = df[df["ClaimNb"] > 0]
        X = d[FEATURES + ["ClaimNb"]]
        y = d["ClaimAmount"] / d["ClaimNb"]
    else:
        raise ValueError("task debe ser 'freq' o 'sev'")
    return X.reset_index(drop=True), y.reset_index(drop=True)


def crear_cv(X: pd.DataFrame, y: pd.Series, task: str, n_splits: int = 5):
    """Folds estratificados por 'hubo siniestro' (evita folds sin positivos)."""
    estrato = (y > 0).astype(int) if task == "freq" else pd.qcut(
        y, q=5, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(X, estrato))


# ──────────────────────────────────────────────────────────────────────
# 3. Preprocesamiento
# ──────────────────────────────────────────────────────────────────────
def construir_preprocesador(kind: str) -> ColumnTransformer:
    if kind == "glm":
        return ColumnTransformer([
            ("num", StandardScaler(), ["VehPower", "LogDensity"]),
            ("bins", KBinsDiscretizer(n_bins=8, encode="onehot-dense",
                                      strategy="quantile"), BIN_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False), CAT_COLS),
        ], remainder="drop")
    return ColumnTransformer([
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ], remainder="drop")


# ──────────────────────────────────────────────────────────────────────
# 4. Modelo envoltorio (peso/exposición + balanceo dentro de cada fold)
# ──────────────────────────────────────────────────────────────────────
class RegresorPonderado(BaseEstimator, RegressorMixin):
    """
    Preprocesa, (opcionalmente) rebalancea y ajusta un regresor con pesos.

    * freq (target_is_count=True): ajusta tasa = ClaimNb/Exposure con
      sample_weight = Exposure (equivale a Poisson con offset log(Exposure)).
      predict() devuelve la TASA anual; los siniestros esperados son tasa*Exposure.
    * sev  (target_is_count=False): ajusta la severidad media con
      sample_weight = ClaimNb. predict() devuelve la severidad media.

    balanceo:
      'sin_balanceo' : nada
      'weights'      : pólizas con siniestro pesan `pos_weight` veces más
      'smote'/'adasyn': sobremuestreo de la clase ClaimNb>0 (solo freq). Las
                        filas sintéticas heredan ClaimNb = media de los positivos
                        y la exposición interpolada. Se aplica SOLO sobre el
                        fold de entrenamiento (nunca sobre validación/test).
    """

    def __init__(self, preprocessor=None, estimator=None,
                 balanceo="sin_balanceo", pos_weight=1.0, sampling_ratio=0.2,
                 weight_col="Exposure", target_is_count=True,
                 random_state=RANDOM_STATE):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.balanceo = balanceo
        self.pos_weight = pos_weight
        self.sampling_ratio = sampling_ratio
        self.weight_col = weight_col
        self.target_is_count = target_is_count
        self.random_state = random_state

    def _separar(self, X):
        w = np.asarray(X[self.weight_col], dtype=float)
        return X.drop(columns=[self.weight_col]), w

    @staticmethod
    def _densa(M):
        return M.toarray() if sparse.issparse(M) else np.asarray(M)

    def _sobremuestrear(self, Xp, y, w):
        clase = (y > 0).astype(int)
        Z = np.column_stack([Xp, w])                 # exposición entra al espacio de vecinos
        if self.balanceo == "smote":
            sam = SMOTE(sampling_strategy=self.sampling_ratio,
                        random_state=self.random_state)
        else:
            sam = ADASYN(sampling_strategy=self.sampling_ratio,
                         random_state=self.random_state)
        Z_res, _ = sam.fit_resample(Z, clase)
        n0 = len(y)
        Xp_res = Z_res[:, :-1]
        w_res = np.clip(Z_res[:, -1], w.min(), w.max())
        y_nuevo = np.full(len(Z_res) - n0, y[y > 0].mean())
        return Xp_res, np.concatenate([y, y_nuevo]), w_res

    def fit(self, X, y):
        Xf, w = self._separar(X)
        y = np.asarray(y, dtype=float)
        self.prep_ = clone(self.preprocessor)
        Xp = self._densa(self.prep_.fit_transform(Xf))
        pesos = w.copy()

        if self.balanceo == "weights":
            pesos = np.where(y > 0, w * self.pos_weight, w)
        elif self.balanceo in ("smote", "adasyn"):
            if not self.target_is_count:
                raise ValueError("SMOTE/ADASYN solo aplican a la frecuencia.")
            n0 = len(y)
            Xp, y, w = self._sobremuestrear(Xp, y, w)
            pesos = np.concatenate([pesos, w[n0:]])
        elif self.balanceo != "sin_balanceo":
            raise ValueError(f"balanceo desconocido: {self.balanceo}")

        objetivo = y / w if self.target_is_count else y
        self.est_ = clone(self.estimator).fit(Xp, objetivo, sample_weight=pesos)
        return self

    def predict(self, X):
        Xf, _ = self._separar(X)
        Xp = self._densa(self.prep_.transform(Xf))
        return np.clip(self.est_.predict(Xp), 1e-9, None)


# ──────────────────────────────────────────────────────────────────────
# 5. Métricas y scorer
# ──────────────────────────────────────────────────────────────────────
def _gini(total_obs, pred, w):
    """Gini de la curva de Lorenz (riesgo ordenado por predicción ascendente)."""
    orden = np.argsort(pred, kind="mergesort")
    cw = np.insert(np.cumsum(w[orden]) / w.sum(), 0, 0.0)
    cy = np.insert(np.cumsum(total_obs[orden]) / total_obs.sum(), 0, 0.0)
    area = np.sum(np.diff(cw) * (cy[1:] + cy[:-1]) / 2.0)
    return float(1.0 - 2.0 * area)


def calcular_metricas(task, y, pred, w):
    """
    freq: y = ClaimNb, pred = tasa anual, w = Exposure
          (deviance/MAE/RMSE sobre siniestros esperados mu = tasa*Exposure)
    sev : y = severidad media, pred = severidad media, w = ClaimNb
          (métricas ponderadas por ClaimNb)
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    w = np.asarray(w, dtype=float)
    if task == "freq":
        mu = pred * w
        return dict(
            deviance=mean_poisson_deviance(y, mu),
            mae=mean_absolute_error(y, mu),
            rmse=float(np.sqrt(mean_squared_error(y, mu))),
            gini=_gini(y, pred, w),
            bias=float(mu.sum() / y.sum()),
        )
    return dict(
        deviance=mean_gamma_deviance(y, pred, sample_weight=w),
        mae=mean_absolute_error(y, pred, sample_weight=w),
        rmse=float(np.sqrt(mean_squared_error(y, pred, sample_weight=w))),
        gini=_gini(y * w, pred, w),
        bias=float((pred * w).sum() / (y * w).sum()),
    )


def crear_scorer(task):
    """Scorer multi-métrica: una sola llamada a predict por fold."""
    wcol = TASKS[task]["weight_col"]

    def scorer(est, X, y):
        m = calcular_metricas(task, y, est.predict(X), X[wcol])
        return {"neg_deviance": -m["deviance"], "mae": m["mae"],
                "rmse": m["rmse"], "gini": m["gini"], "bias": m["bias"]}
    return scorer


# ──────────────────────────────────────────────────────────────────────
# 6. Modelos e hiperparámetros
# ──────────────────────────────────────────────────────────────────────
def obtener_modelos(task: str, usar_gpu: bool = False, threads: int = 1):
    """Devuelve {nombre: (estimador_base, distribuciones, tipo_preprocesador)}."""
    p = "estimator__"
    if task == "freq":
        glm = PoissonRegressor(max_iter=1000)
        xgb_est = xgb.XGBRegressor(objective="count:poisson")
        lgb_est = lgb.LGBMRegressor(objective="poisson")
    else:
        glm = GammaRegressor(max_iter=1000)
        xgb_est = xgb.XGBRegressor(objective="reg:gamma")
        lgb_est = lgb.LGBMRegressor(objective="gamma")

    xgb_est.set_params(tree_method="hist", random_state=RANDOM_STATE, verbosity=0,
                       n_jobs=threads, **({"device": "cuda"} if usar_gpu else {}))
    lgb_est.set_params(random_state=RANDOM_STATE, n_jobs=threads, verbose=-1,
                       subsample_freq=1)

    return {
        "GLM": (glm, {
            p + "alpha": loguniform(1e-7, 1e-1),       # penalización L2
        }, "glm"),
        "XGBoost": (xgb_est, {
            p + "n_estimators": randint(100, 600),
            p + "max_depth": randint(2, 7),
            p + "learning_rate": loguniform(0.01, 0.2),
            p + "subsample": uniform(0.6, 0.4),        # [0.6, 1.0]
            p + "colsample_bytree": uniform(0.5, 0.5), # [0.5, 1.0]
            p + "min_child_weight": loguniform(0.1, 10),
            p + "reg_lambda": loguniform(0.1, 50),
        }, "tree"),
        "LightGBM": (lgb_est, {
            p + "n_estimators": randint(100, 600),
            p + "num_leaves": randint(8, 64),
            p + "learning_rate": loguniform(0.01, 0.2),
            p + "min_child_samples": randint(20, 400),
            p + "subsample": uniform(0.6, 0.4),
            p + "colsample_bytree": uniform(0.5, 0.5),
            p + "reg_lambda": loguniform(0.1, 50),
        }, "tree"),
    }


# ──────────────────────────────────────────────────────────────────────
# 7. Utilidades de guardado
# ──────────────────────────────────────────────────────────────────────
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _escribir_json_atomico(ruta: Path, obj: dict):
    tmp = ruta.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, ruta)


def leer_checkpoints(carpeta="models") -> pd.DataFrame:
    ruta = Path(carpeta) / "checkpoints.csv"
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=CSV_COLS)


def mejores_por_tarea(df_ck: pd.DataFrame) -> pd.DataFrame:
    """Mejor combinación por tarea según la deviance de VALIDACIÓN cruzada."""
    ok = df_ck[df_ck["status"] == "ok"]
    if ok.empty:
        return ok
    return ok.loc[ok.groupby("task")["cv_deviance"].idxmin()].reset_index(drop=True)


def cargar_modelo(task, modelo, balanceo, carpeta="models"):
    return joblib.load(Path(carpeta) / f"{task}__{modelo}__{balanceo}.joblib")


# ──────────────────────────────────────────────────────────────────────
# 8. Entrenamiento de una combinación y bucle con checkpoints
# ──────────────────────────────────────────────────────────────────────
def entrenar_combinacion(task, modelo, balanceo, X_tr, y_tr, X_te, y_te,
                         cv_splits, n_iter=15, n_jobs=-1, usar_gpu=False,
                         sampling_ratio=0.2, backend="loky", threads_modelo=None):
    # búsqueda en serie (n_jobs=1) -> los árboles usan todos los hilos;
    # búsqueda en paralelo -> cada modelo usa 1 hilo (evita sobresuscripción)
    if threads_modelo is None:
        threads_modelo = (os.cpu_count() or 1) if n_jobs == 1 else 1
    est, dist, kind = obtener_modelos(task, usar_gpu, threads_modelo)[modelo]
    dist = dict(dist)
    if balanceo == "weights":
        dist["pos_weight"] = loguniform(1.5, 30)

    cfg = TASKS[task]
    base = RegresorPonderado(
        preprocessor=construir_preprocesador(kind), estimator=est,
        balanceo=balanceo, sampling_ratio=sampling_ratio,
        weight_col=cfg["weight_col"], target_is_count=cfg["target_is_count"],
    )
    rs = RandomizedSearchCV(
        base, dist, n_iter=n_iter, cv=cv_splits,
        scoring=crear_scorer(task), refit="neg_deviance",
        n_jobs=1 if usar_gpu else n_jobs, random_state=RANDOM_STATE,
        error_score="raise", verbose=0,
    )
    t0 = time.time()
    with parallel_config(backend=backend):
        rs.fit(X_tr, y_tr)
    t_busqueda = time.time() - t0

    best = rs.best_estimator_
    i = rs.best_index_
    r = rs.cv_results_
    wcol = cfg["weight_col"]
    m_te = calcular_metricas(task, y_te, best.predict(X_te), X_te[wcol])
    m_tr = calcular_metricas(task, y_tr, best.predict(X_tr), X_tr[wcol])

    rec = dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        task=task, modelo=modelo, balanceo=balanceo,
        n_iter=n_iter, cv_folds=len(cv_splits),
        n_train=len(X_tr), n_test=len(X_te),
        cv_deviance=-r["mean_test_neg_deviance"][i],
        cv_deviance_std=r["std_test_neg_deviance"][i],
        cv_mae=r["mean_test_mae"][i], cv_rmse=r["mean_test_rmse"][i],
        cv_gini=r["mean_test_gini"][i], cv_bias=r["mean_test_bias"][i],
        test_deviance=m_te["deviance"], test_mae=m_te["mae"],
        test_rmse=m_te["rmse"], test_gini=m_te["gini"], test_bias=m_te["bias"],
        train_deviance=m_tr["deviance"],
        tiempo_busqueda_s=round(t_busqueda, 1),
        tiempo_refit_s=round(float(rs.refit_time_), 1),
        fit_time_medio_cv_s=round(float(r["mean_fit_time"][i]), 1),
        best_params=json.dumps(rs.best_params_, default=_json_default),
    )
    return rec, best, rs.best_params_


def ejecutar_experimentos(task, modelos, balanceos, X_tr, y_tr, X_te, y_te,
                          carpeta="models", n_iter=15, cv_folds=5, n_jobs=-1,
                          usar_gpu=False, sampling_ratio=0.2, reanudar=True,
                          backend="loky", threads_modelo=None):
    """
    Recorre modelo x balanceo. Tras CADA combinación:
      - añade una fila a  {carpeta}/checkpoints.csv
      - actualiza         {carpeta}/mejores_modelos.json
      - guarda el modelo  {carpeta}/{task}__{modelo}__{balanceo}.joblib
    Si reanudar=True se omiten las combinaciones ya registradas con status 'ok'.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta / "checkpoints.csv"
    ruta_json = carpeta / "mejores_modelos.json"

    ck = leer_checkpoints(carpeta)
    hechos = set(map(tuple, ck.loc[ck["status"] == "ok", ["task", "modelo", "balanceo"]]
                     .values)) if reanudar else set()

    if ruta_json.exists():
        with open(ruta_json, encoding="utf-8") as f:
            js = json.load(f)
    else:
        js = {"meta": {}, "resultados": {}, "mejor_por_tarea": {}}
    js["meta"] = dict(
        actualizado=datetime.now().isoformat(timespec="seconds"),
        random_state=RANDOM_STATE, python=platform.python_version(),
        sklearn=sklearn.__version__, xgboost=xgb.__version__,
        lightgbm=lgb.__version__,
    )

    cv_splits = crear_cv(X_tr, y_tr, task, cv_folds)
    total = len(modelos) * len(balanceos)
    k = 0
    for modelo in modelos:
        for bal in balanceos:
            k += 1
            clave = f"{task}|{modelo}|{bal}"
            if (task, modelo, bal) in hechos:
                print(f"[{k}/{total}] {clave}  -> ya existe en checkpoints, se omite")
                continue
            print(f"[{k}/{total}] {clave} ...", flush=True)
            try:
                rec, best, params = entrenar_combinacion(
                    task, modelo, bal, X_tr, y_tr, X_te, y_te, cv_splits,
                    n_iter=n_iter, n_jobs=n_jobs, usar_gpu=usar_gpu,
                    sampling_ratio=sampling_ratio, backend=backend,
                    threads_modelo=threads_modelo)
                ruta_modelo = carpeta / f"{task}__{modelo}__{bal}.joblib"
                joblib.dump(best, ruta_modelo, compress=3)
                rec.update(modelo_path=str(ruta_modelo), status="ok", error="")
                print(f"      CV dev={rec['cv_deviance']:.5f} | "
                      f"test dev={rec['test_deviance']:.5f} | "
                      f"MAE={rec['test_mae']:.5f} | RMSE={rec['test_rmse']:.5f} | "
                      f"Gini={rec['test_gini']:.3f} | sesgo={rec['test_bias']:.3f} | "
                      f"{rec['tiempo_busqueda_s']}s")
                js["resultados"][clave] = dict(
                    best_params=params,
                    cv={x: rec[f"cv_{x}"] for x in
                        ["deviance", "deviance_std", "mae", "rmse", "gini", "bias"]},
                    test={x: rec[f"test_{x}"] for x in
                          ["deviance", "mae", "rmse", "gini", "bias"]},
                    train_deviance=rec["train_deviance"],
                    tiempos_s=dict(busqueda=rec["tiempo_busqueda_s"],
                                   refit=rec["tiempo_refit_s"],
                                   fit_medio_cv=rec["fit_time_medio_cv_s"]),
                    n_iter=n_iter, cv_folds=cv_folds,
                    modelo_path=str(ruta_modelo), timestamp=rec["timestamp"],
                )
            except Exception as e:                      # no se pierde lo ya guardado
                rec = {c: np.nan for c in CSV_COLS}
                rec.update(timestamp=datetime.now().isoformat(timespec="seconds"),
                           task=task, modelo=modelo, balanceo=bal,
                           status="error", error=repr(e)[:300])
                print(f"      ERROR: {e!r}")
                traceback.print_exc()

            pd.DataFrame([rec])[CSV_COLS].to_csv(
                ruta_csv, mode="a", header=not ruta_csv.exists(), index=False)

            # mejor por tarea (según CV, no según test)
            ok = {c: v for c, v in js["resultados"].items() if c.startswith(task + "|")}
            if ok:
                mejor = min(ok, key=lambda c: ok[c]["cv"]["deviance"])
                js["mejor_por_tarea"][task] = dict(clave=mejor, **ok[mejor])
            _escribir_json_atomico(ruta_json, js)

    return leer_checkpoints(carpeta)


# ──────────────────────────────────────────────────────────────────────
# 9. Prima pura (opcional): frecuencia x severidad sobre el test
# ──────────────────────────────────────────────────────────────────────
def evaluar_prima_pura(est_freq, est_sev, df_test: pd.DataFrame, power=1.5):
    """Prima pura = tasa*Exposure * severidad media, contra ClaimAmount real."""
    X_f, _ = armar_xy(df_test, "freq")
    tasa = est_freq.predict(X_f)
    X_s = df_test[FEATURES].assign(ClaimNb=1.0)   # el peso se ignora al predecir
    sev = est_sev.predict(X_s)
    expo = df_test["Exposure"].to_numpy()
    pp = tasa * expo * sev
    obs = df_test["ClaimAmount"].to_numpy()
    return dict(
        mae=mean_absolute_error(obs, pp),
        rmse=float(np.sqrt(mean_squared_error(obs, pp))),
        tweedie_dev=mean_tweedie_deviance(obs, pp, power=power),
        gini=_gini(obs, pp / expo, expo),
        bias=float(pp.sum() / obs.sum()),
    )