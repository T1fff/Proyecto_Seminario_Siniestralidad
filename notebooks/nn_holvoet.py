"""
notebooks_06neuralnetworks/nn_holvoet.py
Replicación en Python (Keras / TensorFlow) de la metodología de:
  Holvoet, Antonio y Henckaerts (2023). Neural Networks for Insurance Pricing with
  Frequency and Severity Data. arXiv:2310.12671
  Código de referencia en R: github.com/freekholvoet/NNforFreqSevPricing

Componentes (siguiendo el código R de los autores cuando el paper no es explícito):
- Datos FR (freMTPL2 del repo). Mismo tratamiento que NNforFS_1_Data_Setup.R:
  Density -> log; DrivAge y VehAge en categorías; VehPower como categórica.
  Continuas escaladas: BonusMalus y LogDensity (media y sd del train).
- Partición: 6 pliegues externos. Dentro de cada pliegue externo:
  búsqueda aleatoria de hiperparámetros con CV interna, y reentrenamiento
  con 20 % de validación para early stopping, repetido R veces.
- Embedding de las categóricas: autoencoder (encoder lineal de dimensión d,
  decoder softmax por variable). Los pesos del encoder se reescalan con la
  media y sd de cada dimensión latente en el train (Ecs. 9–10) y después se
  afinan durante el entrenamiento supervisado.
- FFNN: capas densas con activación h, dropout, salida exponencial.
- CANN (Ec. 4): salida = exp(w_NN * ŷ_NN + w_IN * ln ŷ_IN + b).
  Fija: w_NN = w_IN = 1, b = 0 (no entrenables). Flexible: todos entrenables.
- Pérdidas: Poisson para frecuencia (la exposición multiplica la tasa, Ec. 1),
  Gamma para severidad, ponderada por ClaimNb (Ec. 2).
- Optimizador Adam, lr por defecto de Keras (0.001), 500 épocas máximo,
  paciencia 20 en early stopping.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import keras
from keras import callbacks, layers, optimizers, Model
from sklearn.metrics import (mean_absolute_error, mean_gamma_deviance,
                             mean_poisson_deviance, mean_squared_error)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import time

RANDOM_STATE = 42


class Progreso:
    """Avance por unidad de entrenamiento y tiempo restante estimado (promedio de unidades ya hechas)."""

    def __init__(self, total):
        self.total = total
        self.hechos = 0
        self.inicio = time.perf_counter()
        self.tiempos = []

    def paso(self, descripcion, segundos):
        self.hechos += 1
        self.tiempos.append(segundos)
        transcurrido = time.perf_counter() - self.inicio
        restante = np.mean(self.tiempos) * (self.total - self.hechos)
        print(f"[{self.hechos:>4}/{self.total}] {100 * self.hechos / self.total:5.1f}% | {descripcion} | "
              f"{segundos:6.1f} s | transcurrido {_hms(transcurrido)} | faltan ~{_hms(restante)}",
              flush=True)


class ProgresoEpocas(callbacks.Callback):
    """Imprime cada `cada` épocas el val_loss y el tiempo transcurrido del entrenamiento."""

    def __init__(self, etiqueta="", cada=10):
        super().__init__()
        self.etiqueta = etiqueta
        self.cada = cada

    def on_train_begin(self, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.cada == 0:
            logs = logs or {}
            val = logs.get("val_loss", logs.get("loss", float("nan")))
            print(f"      {self.etiqueta} época {epoch + 1} | val_loss {val:.5f} | "
                  f"{time.perf_counter() - self.t0:6.1f} s", flush=True)


def _hms(seg):
    seg = int(seg)
    return f"{seg // 3600}:{(seg % 3600) // 60:02d}:{seg % 60:02d}"

# Variables como en NNforFS_1_Data_Setup.R (cat_FR y scale_FR)
CAT_FR = ["Area", "VehPower", "VehBrand", "Region", "VehGas", "VehAge", "DrivAge"]
NUM_FR = ["BonusMalus", "LogDensity"]

# Espacio de búsqueda de la Tabla 3 del paper / NNforFS_4 (activación oculta, batch, capas, nodos, dropout)
RANGOS = {
    "freq": dict(batch=(10_000, 50_000)),
    "sev": dict(batch=(200, 10_000)),
}
ACTIVACIONES = ["relu", "sigmoid", "softmax"]
CAPAS = (1, 4)
NODOS = (10, 50)
DROPOUT = (1e-4, 0.1)
MAX_EPOCHS = 500
PACIENCIA = 20


# ──────────────────────────────────────────────────────────────────────
# 1. Datos
# ──────────────────────────────────────────────────────────────────────
def preparar_fr(df: pd.DataFrame) -> pd.DataFrame:
    """df = preparar_datos(freq, sev) del repo. Aplica el tratamiento FR del R original."""
    d = df.copy()
    d["LogDensity"] = np.log(d["Density"].clip(lower=1))
    d["DrivAge"] = pd.cut(d["DrivAge"], [-np.inf, 21, 26, 30, 40, 50, 70, np.inf],
                          right=False, labels=False) + 1
    d["VehAge"] = np.select([d["VehAge"] == 0, d["VehAge"] < 11], [1, 2], default=3)
    d["average"] = np.where(d["ClaimNb"] > 0, d["ClaimAmount"] / d["ClaimNb"].clip(lower=1), np.nan)
    return d.reset_index(drop=True)


def asignar_pliegues(d: pd.DataFrame, k: int = 6, seed: int = RANDOM_STATE) -> np.ndarray:
    """Pliegues estratificados por ClaimNb>0 (texto del paper)."""
    pliegue = np.zeros(len(d), dtype=int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(skf.split(d, (d["ClaimNb"] > 0).astype(int)), 1):
        pliegue[te] = i
    return pliegue


class Preprocesamiento:
    """Ajustado sólo con el train de cada pliegue externo (sin fuga)."""

    def fit(self, d: pd.DataFrame):
        self.mu = d[NUM_FR].mean()
        self.sd = d[NUM_FR].std()
        self.oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.oh.fit(d[CAT_FR].astype(str))
        self.bloques = [len(c) for c in self.oh.categories_]
        return self

    def numericas(self, d):
        return ((d[NUM_FR] - self.mu) / self.sd).to_numpy("float32")

    def categoricas(self, d):
        return self.oh.transform(d[CAT_FR].astype(str)).astype("float32")


# ──────────────────────────────────────────────────────────────────────
# 2. Autoencoder de embeddings (Ecs. 6–10)
# ──────────────────────────────────────────────────────────────────────
def entrenar_autoencoder(Z, bloques, d_latente, max_epochs=MAX_EPOCHS, seed=RANDOM_STATE):
    """Encoder lineal + decoder softmax por variable. Pérdida: entropía cruzada por bloque."""
    keras.utils.set_random_seed(seed)
    entrada = layers.Input((Z.shape[1],))
    enc = layers.Dense(d_latente, activation="linear", name="encoder")
    latente = enc(entrada)
    salidas, inicio = [], 0
    for L in bloques:
        salidas.append(layers.Dense(L, activation="softmax")(latente))
        inicio += L
    ae = Model(entrada, salidas)
    ae.compile(optimizer=optimizers.Nadam(), loss=["binary_crossentropy"] * len(bloques))
    objetivos, inicio = [], 0
    for L in bloques:
        objetivos.append(Z[:, inicio:inicio + L])
        inicio += L
    ae.fit(Z, objetivos, epochs=max_epochs, batch_size=1000, validation_split=0.2, verbose=0,
           callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                                              restore_best_weights=True)])
    W, b = enc.get_weights()
    return W, b


def reescalar_encoder(W, b, Z):
    """Ecs. 9–10: divide pesos por sd y ajusta el sesgo con la media de cada dimensión latente (train)."""
    lat = Z @ W + b
    mu, sd = lat.mean(axis=0), lat.std(axis=0, ddof=1)
    return (W / sd).astype("float32"), ((b - mu) / sd).astype("float32")


# ──────────────────────────────────────────────────────────────────────
# 3. Arquitecturas
# ──────────────────────────────────────────────────────────────────────
def gamma_deviance_loss(y_true, y_pred):
    """Deviance gamma por observación (la ponderación por ClaimNb entra por sample_weight)."""
    return 2.0 * ((y_true - y_pred) / y_pred - keras.ops.log(y_true / y_pred))


def construir_red(tarea, n_num, n_oh, config, pesos_ae, cann=False, cann_fija=True):
    """
    FFNN o CANN para 'freq' (Poisson, con exposición) o 'sev' (Gamma).
    pesos_ae: (W, b) reescalados del encoder de categóricas.
    """
    x_num = layers.Input((n_num,), name="num")
    x_oh = layers.Input((n_oh,), name="oh")
    x_expo = layers.Input((1,), name="expo")
    entradas = [x_num, x_oh]
    if tarea == "freq":
        entradas.append(x_expo)

    emb = layers.Dense(pesos_ae[0].shape[1], name="emb_ae")
    emb_out = emb(x_oh)
    emb.set_weights([pesos_ae[0], pesos_ae[1]])
    h = layers.Concatenate()([x_num, emb_out])

    for q, r in zip(config["capas_nodos"], config["dropout"]):
        h = layers.Dense(q, activation=config["activacion"])(h)
        if r > 0:
            h = layers.Dropout(r)(h)

    if cann:
        x_in = layers.Input((1,), name="log_glm")
        entradas.append(x_in)
        pre = layers.Dense(1, activation="linear", name="nn_pre")(h)    # ŷ_NN (escala log)
        comb = layers.Concatenate()([pre, x_in])                         # [ŷ_NN, ln ŷ_IN]
        salida = layers.Dense(1, activation="exponential", name="cann_out", trainable=not cann_fija)
        tasa = salida(comb)
        salida.set_weights([np.array([[1.0], [1.0]], dtype="float32"), np.zeros(1, dtype="float32")])
    else:
        tasa = layers.Dense(1, activation="exponential", name="out")(h)

    if tarea == "freq":
        mu = layers.Multiply()([tasa, x_expo])          # Ec. 1: e_i · λ_i
        modelo = Model(entradas, mu)
        modelo.compile(optimizer=optimizers.Adam(), loss="poisson")
    else:
        modelo = Model(entradas, tasa)
        modelo.compile(optimizer=optimizers.Adam(), loss=gamma_deviance_loss)
    return modelo


def entradas_de(P, d, tarea, glm_pred=None):
    """Arma la lista de entradas del modelo en el orden de construir_red."""
    x = [P.numericas(d), P.categoricas(d)]
    if tarea == "freq":
        x.append(d["Exposure"].to_numpy("float32").reshape(-1, 1))
    if glm_pred is not None:
        x.append(np.log(glm_pred).astype("float32").reshape(-1, 1))
    return x


def objetivo_de(d, tarea):
    if tarea == "freq":
        return d["ClaimNb"].to_numpy("float32")
    return d["average"].to_numpy("float32")


def pesos_muestra(d, tarea):
    return np.ones(len(d), dtype="float32") if tarea == "freq" else d["ClaimNb"].to_numpy("float32")


# ──────────────────────────────────────────────────────────────────────
# 4. Entrenamiento, búsqueda y predicción
# ──────────────────────────────────────────────────────────────────────
def muestrear_configuraciones(n, tarea, rng):
    lo_b, hi_b = RANGOS[tarea]["batch"]
    configs = []
    for _ in range(n):
        capas = int(round(np.exp(rng.uniform(np.log(CAPAS[0]), np.log(CAPAS[1])))))
        nodos = int(round(np.exp(rng.uniform(np.log(NODOS[0]), np.log(NODOS[1])))))
        configs.append(dict(
            batch=int(round(np.exp(rng.uniform(np.log(lo_b), np.log(hi_b))))),
            activacion=str(rng.choice(ACTIVACIONES)),
            capas_nodos=[nodos] * capas,
            dropout=[round(float(np.exp(rng.uniform(np.log(DROPOUT[0]), np.log(DROPOUT[1])))), 4)] * capas,
        ))
    return configs


def deviance_media(tarea, y, mu, w):
    if tarea == "freq":
        return mean_poisson_deviance(y, mu)
    return mean_gamma_deviance(y, mu, sample_weight=w)


def ajustar(tarea, config, X, y, w, pesos_ae, cann, cann_fija, validacion, max_epochs,
            seed, etiqueta=""):
    """validacion: ('split', 0.2) o ('datos', (Xv, yv, wv)). Early stopping sobre val_loss."""
    keras.utils.set_random_seed(seed)
    n_oh = X[1].shape[1]
    modelo = construir_red(tarea, X[0].shape[1], n_oh, config, pesos_ae, cann, cann_fija)
    kw = {}
    if validacion[0] == "split":
        kw["validation_split"] = validacion[1]
    else:
        kw["validation_data"] = (validacion[1][0], validacion[1][1], validacion[1][2])
    modelo.fit(X, y, sample_weight=w, epochs=max_epochs, batch_size=config["batch"],
               verbose=0, shuffle=True,
               callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                                                  restore_best_weights=True),
                          ProgresoEpocas(etiqueta)],
               **kw)
    return modelo


def mu_predicho(modelo, X):
    return modelo.predict(X, batch_size=65536, verbose=0).reshape(-1)


def buscar_hiperparametros(tarea, d_tr, P, pesos_ae, cann, glm_tr, n_configs, pliegues_int,
                           max_epochs, rng, log=print, progreso=None, etiqueta=""):
    """CV interna: devuelve la configuración con menor deviance media de validación."""
    configs = muestrear_configuraciones(n_configs, tarea, rng)
    y = objetivo_de(d_tr, tarea)
    w = pesos_muestra(d_tr, tarea)
    X_all = entradas_de(P, d_tr, tarea, glm_tr)
    mejor, mejor_dev = None, np.inf
    pliegues_unicos = np.unique(pliegues_int)
    for c_id, cfg in enumerate(configs):
        devs = []
        for j in pliegues_unicos:
            t0 = time.perf_counter()
            tr = np.where(pliegues_int != j)[0]
            va = np.where(pliegues_int == j)[0]
            Xtr = [x[tr] for x in X_all]
            Xva = [x[va] for x in X_all]
            m = ajustar(tarea, cfg, Xtr, y[tr], w[tr], pesos_ae, cann, True,
                        ("datos", (Xva, y[va], w[va])), max_epochs, seed=int(RANDOM_STATE + j),
                        etiqueta=f"{etiqueta} CV{int(j)}")
            devs.append(deviance_media(tarea, y[va], mu_predicho(m, Xva), w[va]))
            if progreso is not None:
                progreso.paso(f"{etiqueta} | cfg {c_id + 1}/{len(configs)} | CV {int(j)}/{len(pliegues_unicos)}",
                              time.perf_counter() - t0)
        dev = float(np.mean(devs))
        log(f"    cfg {c_id}: {cfg['activacion']}, capas={cfg['capas_nodos']}, "
            f"batch={cfg['batch']}, dropout={cfg['dropout'][0]} -> dev CV={dev:.5f}")
        if dev < mejor_dev:
            mejor, mejor_dev = cfg, dev
    return mejor, mejor_dev, configs


def entrenar_final(tarea, cfg, d_tr, P, pesos_ae, cann, glm_tr, glm_te, d_te, repeticiones,
                   max_epochs, cann_fija=True, progreso=None, etiqueta=""):
    """Reentrena R veces con 20 % de validación para early stopping. Promedia predicciones."""
    y = objetivo_de(d_tr, tarea)
    w = pesos_muestra(d_tr, tarea)
    X_tr = entradas_de(P, d_tr, tarea, glm_tr)
    X_te = entradas_de(P, d_te, tarea, glm_te)
    preds = []
    for r in range(repeticiones):
        t0 = time.perf_counter()
        m = ajustar(tarea, cfg, X_tr, y, w, pesos_ae, cann, cann_fija, ("split", 0.2),
                    max_epochs, seed=int(RANDOM_STATE + 100 + r), etiqueta=f"{etiqueta} final{r + 1}")
        preds.append(mu_predicho(m, X_te))
        if progreso is not None:
            progreso.paso(f"{etiqueta} | entrenamiento final {r + 1}/{repeticiones}",
                          time.perf_counter() - t0)
    return np.mean(preds, axis=0), len(preds)


# ──────────────────────────────────────────────────────────────────────
# 5. GLM de referencia (y ŷ_IN para CANN)
# ──────────────────────────────────────────────────────────────────────
def matriz_glm(P, d):
    """Constante + numéricas escaladas + dummies de categóricas (se descarta la primera)."""
    cats = pd.get_dummies(d[CAT_FR].astype(str), drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(d)), P.numericas(d), cats.to_numpy()])
    return X


def glm_predict(d_tr, d_te, tarea, P):
    """Freq: Poisson log-link con offset log(expo). Sev: Gamma log-link con pesos ClaimNb."""
    if tarea == "freq":
        Xtr = matriz_glm(P, d_tr)
        Xte = matriz_glm(P, d_te)
        m = sm.GLM(d_tr["ClaimNb"].to_numpy(float), Xtr, family=sm.families.Poisson(),
                   offset=np.log(d_tr["Exposure"].to_numpy(float))).fit()
        tasa_te = np.exp(Xte @ m.params)
        tasa_tr = np.exp(Xtr @ m.params)
        return tasa_tr, tasa_te
    dtr = d_tr[d_tr["ClaimNb"] > 0]
    Xtr = matriz_glm(P, dtr)
    m = sm.GLM(dtr["average"].to_numpy(float), Xtr, family=sm.families.Gamma(sm.families.links.Log()),
               var_weights=dtr["ClaimNb"].to_numpy(float)).fit()
    return np.exp(matriz_glm(P, d_tr) @ m.params), np.exp(matriz_glm(P, d_te) @ m.params)


# ──────────────────────────────────────────────────────────────────────
# 6. Métricas
# ──────────────────────────────────────────────────────────────────────
def metricas(tarea, y, mu, w):
    if tarea == "freq":
        return dict(deviance=mean_poisson_deviance(y, mu), mae=mean_absolute_error(y, mu),
                    rmse=float(np.sqrt(mean_squared_error(y, mu))))
    return dict(deviance=mean_gamma_deviance(y, mu, sample_weight=w),
                mae=mean_absolute_error(y, mu, sample_weight=w),
                rmse=float(np.sqrt(mean_squared_error(y, mu, sample_weight=w))))
