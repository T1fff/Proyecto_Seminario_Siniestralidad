"""
notebooks/nn_clases.py
Severidad en tres bandas (barato, intermedio, caro) con redes neuronales.

Reutiliza de nn_holvoet.py: preprocesamiento de FR, autoencoder de embeddings,
el muestreo del espacio de búsqueda (Tabla 3 del paper) y la clase Progreso.
Reutiliza de src/clasificacion.py: los cinco métodos de corte, el etiquetado, la GVF y las métricas.

Diferencias respecto a la FFNN de regresión de nn_holvoet.py:
- Salida softmax con K = 3 neuronas y entropía cruzada, en lugar de salida exponencial con pérdida Gamma.
- Sin CANN: su salida exponencial sólo tiene sentido en regresión.
- Sin ponderación por ClaimNb: cada póliza con siniestro cuenta una vez, porque la banda se asigna a la
  severidad media de la póliza.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time

import keras
import numpy as np
from keras import callbacks, layers, optimizers, Model
from sklearn.linear_model import LogisticRegression

from nn_holvoet import (PACIENCIA, RANDOM_STATE, ProgresoEpocas, matriz_glm,
                        muestrear_configuraciones)
from src.clasificacion import K, etiquetar
from src.clasificacion import metricas as metricas_clasificacion


def log_monto(d):
    """Respuesta para los cortes: log1p de la severidad media por siniestro de cada póliza."""
    return np.log1p(d["average"].to_numpy(dtype=float))


def entradas_clases(P, d):
    """Entradas de la red: numéricas escaladas y dummies de las categóricas (el embedding va dentro)."""
    return [P.numericas(d), P.categoricas(d)]


def construir_red_clases(n_num, n_oh, config, pesos_ae):
    """Igual que construir_red de nn_holvoet para la rama de embeddings, con salida softmax de K clases."""
    x_num = layers.Input((n_num,), name="num")
    x_oh = layers.Input((n_oh,), name="oh")

    emb = layers.Dense(pesos_ae[0].shape[1], name="emb_ae")
    emb_out = emb(x_oh)
    emb.set_weights([pesos_ae[0], pesos_ae[1]])
    h = layers.Concatenate()([x_num, emb_out])

    for q, r in zip(config["capas_nodos"], config["dropout"]):
        h = layers.Dense(q, activation=config["activacion"])(h)
        if r > 0:
            h = layers.Dropout(r)(h)

    salida = layers.Dense(K, activation="softmax", name="bandas")(h)
    modelo = Model([x_num, x_oh], salida)
    modelo.compile(optimizer=optimizers.Adam(), loss="sparse_categorical_crossentropy")
    return modelo


def ajustar_clases(config, X, y, pesos_ae, max_epochs, seed, validacion, etiqueta=""):
    """validacion: ('split', 0.2) o ('datos', (Xv, yv)). Early stopping sobre val_loss."""
    keras.utils.set_random_seed(seed)
    modelo = construir_red_clases(X[0].shape[1], X[1].shape[1], config, pesos_ae)
    kw = {}
    if validacion[0] == "split":
        kw["validation_split"] = validacion[1]
    else:
        kw["validation_data"] = (validacion[1][0], validacion[1][1])
    modelo.fit(X, y, epochs=max_epochs, batch_size=config["batch"], verbose=0, shuffle=True,
               callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                                                  restore_best_weights=True),
                          ProgresoEpocas(etiqueta)],
               **kw)
    return modelo


def buscar_config_clases(metodo, d_tr, log_tr, P, pesos_ae, funciones, n_configs, pliegues_int,
                         max_epochs, rng, progreso=None, etiqueta=""):
    """
    CV interna. En cada pliegue interno los cortes se estiman sólo con su train interno,
    y se mide el F1 macro en su validación. Devuelve la configuración con mayor F1 medio.
    """
    configs = muestrear_configuraciones(n_configs, "sev", rng)
    X_all = entradas_clases(P, d_tr)
    mejor, mejor_f1 = None, -np.inf
    pliegues_unicos = np.unique(pliegues_int)
    for c_id, cfg in enumerate(configs):
        f1s = []
        for j in pliegues_unicos:
            t0 = time.perf_counter()
            tr = np.where(pliegues_int != j)[0]
            va = np.where(pliegues_int == j)[0]
            cortes = funciones[metodo](log_tr[tr])
            y_all = etiquetar(log_tr, cortes)
            Xtr = [x[tr] for x in X_all]
            Xva = [x[va] for x in X_all]
            m = ajustar_clases(cfg, Xtr, y_all[tr], pesos_ae, max_epochs,
                               seed=int(RANDOM_STATE + j), validacion=("datos", (Xva, y_all[va])),
                               etiqueta=f"{etiqueta} CV{int(j)}")
            yhat = m.predict(Xva, batch_size=65536, verbose=0).argmax(axis=1)
            f1s.append(metricas_clasificacion(y_all[va], yhat)["f1_macro"])
            if progreso is not None:
                progreso.paso(f"{etiqueta} | cfg {c_id + 1}/{len(configs)} | CV {int(j)}/{len(pliegues_unicos)}",
                              time.perf_counter() - t0)
        f1_medio = float(np.mean(f1s))
        print(f"    cfg {c_id}: {cfg['activacion']}, capas={cfg['capas_nodos']}, "
              f"batch={cfg['batch']}, dropout={cfg['dropout'][0]} -> F1 CV={f1_medio:.4f}", flush=True)
        if f1_medio > mejor_f1:
            mejor, mejor_f1 = cfg, f1_medio
    return mejor, mejor_f1, configs


def entrenar_final_clases(cfg, X_tr, y_tr, X_te, pesos_ae, max_epochs, repeticiones,
                          progreso=None, etiqueta=""):
    """Reentrena con 20 % de validación para early stopping. Promedia probabilidades y devuelve la clase."""
    probs = []
    for r in range(repeticiones):
        t0 = time.perf_counter()
        m = ajustar_clases(cfg, X_tr, y_tr, pesos_ae, max_epochs, seed=int(RANDOM_STATE + 100 + r),
                           validacion=("split", 0.2), etiqueta=f"{etiqueta} final{r + 1}")
        probs.append(m.predict(X_te, batch_size=65536, verbose=0))
        if progreso is not None:
            progreso.paso(f"{etiqueta} | entrenamiento final {r + 1}/{repeticiones}",
                          time.perf_counter() - t0)
    return np.mean(probs, axis=0).argmax(axis=1)


def glm_clases(P, d_tr, y_tr, d_te):
    """Referencia: regresión logística multinomial sobre las mismas variables que la red (sin embeddings)."""
    Xtr = matriz_glm(P, d_tr)[:, 1:]          # se quita la constante: LogisticRegression ya tiene intercepto
    Xte = matriz_glm(P, d_te)[:, 1:]
    m = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr, y_tr)
    return m.predict(Xte)
