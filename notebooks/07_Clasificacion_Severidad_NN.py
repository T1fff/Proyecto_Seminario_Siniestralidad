# %% [markdown]
# # Severidad en tres bandas con redes neuronales (barato, intermedio, caro)
#
# Versión del notebook 06 en la que el modelo de severidad clasifica la severidad media por siniestro
# en tres bandas, en lugar de predecirla. Usa los cinco métodos de corte del notebook 05
# (`src/clasificacion.py`) y la misma estructura de pliegues, búsqueda de hiperparámetros, progreso y ejecución en paralelo.
#
# **Datos:** `freMTPL2freq.csv` + `freMTPL2sev.csv`, igual que el 05 y el 06.
# **Respuesta:** severidad media por siniestro (`ClaimAmount / ClaimNb`), sólo pólizas con siniestro.
# **Métodos de corte:** Jenks, k-means, regla de Wang (con extensión propia en la cola inferior), ancho igual en log
# y cortes fijos. Los cortes se estiman sólo con el train de cada pliegue, y dentro de cada pliegue interno.
# **Modelos:** FFNN con salida softmax y GLM (regresión logística multinomial). La CANN se omite porque
# requiere salida de regresión.
#
# **Desviaciones del paper** (además de las del 06): salida softmax con entropía cruzada, sin CANN, y sin
# ponderación por ClaimNb.

# %% [markdown]
# ## 0. Configuración

# %%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
_hilos = int(os.environ.get("TF_THREADS", "0"))   # 0 = todos; en paralelo, 2 por proceso
tf.config.threading.set_intra_op_parallelism_threads(_hilos)
tf.config.threading.set_inter_op_parallelism_threads(1 if _hilos else 0)
print("Dispositivo: CPU (sin GPU Metal)" if not tf.config.list_physical_devices("GPU") else "Dispositivo: GPU")

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent))

from src.utils import load_freq, load_sev
from src.modelado import preparar_datos
from src.clasificacion import ETIQUETAS, K, METODOS, gvf, obtener_funciones_cortes, etiquetar
from nn_holvoet import (preparar_fr, asignar_pliegues, Preprocesamiento, entrenar_autoencoder,
                        reescalar_encoder, RANDOM_STATE, Progreso)
from nn_clases import (log_monto, entradas_clases, buscar_config_clases, entrenar_final_clases,
                       glm_clases, metricas_clasificacion)

# Cortes fijos en unidades de monto, igual que el 05. Hay que confirmarlos con criterio de negocio.
CORTES_FIJOS_MONTO = (1000.0, 3000.0)
FUNCIONES = obtener_funciones_cortes(cortes_monto=CORTES_FIJOS_MONTO)

MODO_PRUEBA = os.environ.get("NN_PRUEBA") == "1"
if MODO_PRUEBA:
    CARPETA = HERE / "resultados_clases_prueba"
    N_CONFIGS, INNER_FOLDS, REPETICIONES, MAX_EPOCHS, AE_EPOCHS = 2, 2, 1, 3, 3
    PLIEGUES_EXTERNOS = [1]
else:
    CARPETA = HERE / "resultados_clases"
    N_CONFIGS, INNER_FOLDS, REPETICIONES, MAX_EPOCHS, AE_EPOCHS = 4, 3, 1, 50, 50
    PLIEGUES_EXTERNOS = [int(x) for x in os.environ.get("PLIEGUES", "1,2,3,4,5,6").split(",")]
N_PLIEGUES = 6
DIM_AE = 15
CARPETA.mkdir(parents=True, exist_ok=True)

# Unidades de progreso por pliegue: autoencoder + por método (GLM + CV interna + repeticiones finales)
UNIDADES_POR_PLIEGUE = 1 + len(METODOS) * (1 + N_CONFIGS * INNER_FOLDS + REPETICIONES)
PROGRESO = Progreso(total=len(PLIEGUES_EXTERNOS) * UNIDADES_POR_PLIEGUE)
print(f"Unidades de entrenamiento: {PROGRESO.total}")

# %% [markdown]
# ## 1. Datos y pliegues
#
# Los pliegues son los mismos que en el 06 (estratificados por `ClaimNb > 0`). La severidad usa sólo las
# pólizas con siniestro, con el pliegue de su póliza.

# %%
df = preparar_datos(load_freq("freMTPL2freq.csv"), load_sev("freMTPL2sev.csv"))
d = preparar_fr(df)
d["pliegue"] = asignar_pliegues(d, k=N_PLIEGUES, seed=RANDOM_STATE)
print(f"Pólizas: {len(d):,} | con siniestro: {(d.ClaimNb > 0).sum():,}")
print(d.groupby("pliegue").size().rename("pólizas").to_frame().T.to_string())

# %% [markdown]
# ## 2a. Entrenamiento en paralelo
#
# Lanza tres procesos (2 hilos de TensorFlow cada uno) y muestra el avance aquí. Los pliegues que ya tienen
# `pliegue_k.json` se omiten. Con `PARALELO = False` no se lanza nada aquí y la celda siguiente corre en secuencia.

# %%
import subprocess
from IPython.display import clear_output

PARALELO = True            # False: no lanza procesos; la celda siguiente corre en secuencia
GRUPOS_PARALELOS = ["1,2", "3,4", "5,6"]

# Guarda: el propio script, al ejecutarse en paralelo, tiene la variable PLIEGUES y no vuelve a lanzar procesos.
if PARALELO and "PLIEGUES" not in os.environ and not MODO_PRUEBA:
    CARPETA_LOGS = HERE / "logs_clases"
    CARPETA_LOGS.mkdir(exist_ok=True)
    procesos = []
    for g in GRUPOS_PARALELOS:
        log = CARPETA_LOGS / f"pliegues_{g.replace(',', '_')}.log"
        env = {**os.environ, "PLIEGUES": g, "TF_THREADS": "2"}
        p = subprocess.Popen([sys.executable, str(HERE / "07_Clasificacion_Severidad_NN.py")],
                             cwd=str(HERE), env=env,
                             stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
        procesos.append((g, log, p))

    inicio = time.time()
    while any(p.poll() is None for _, _, p in procesos):
        clear_output(wait=True)
        print(f"Transcurrido: {int(time.time() - inicio)} s")
        for g, log, p in procesos:
            lineas = [l for l in log.read_text(encoding="utf-8", errors="ignore").splitlines()
                      if l.strip().startswith("[")]
            if p.poll() is not None:
                estado = "terminado"
            else:
                estado = lineas[-1].strip() if lineas else "iniciando..."
            print(f"pliegues {g}: {estado}")
        time.sleep(30)

    clear_output(wait=True)
    print(f"Transcurrido total: {int(time.time() - inicio)} s")
    for g, log, p in procesos:
        print(f"pliegues {g}: código de salida {p.returncode} (log: logs_clases/{log.name})")

# %% [markdown]
# ## 2. Entrenamiento por pliegue externo
#
# Por cada pliegue externo: (1) preprocesamiento y autoencoder sólo con el train; (2) para cada método de corte,
# los cortes se estiman con el train del pliegue, se entrena un GLM de referencia y una FFNN con búsqueda aleatoria
# de hiperparámetros. Cada pliegue se guarda en `resultados_clases/pliegue_k.json`, así que la corrida puede reanudarse.

# %%
def correr_pliegue(k):
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_STATE + k)
    tr_pol = d[d.pliegue != k].reset_index(drop=True)
    P = Preprocesamiento().fit(tr_pol)

    # Autoencoder de embeddings (no supervisado, con todas las pólizas del train, como en el 06)
    Z = P.categoricas(tr_pol)
    t_ae = time.perf_counter()
    W, b = entrenar_autoencoder(Z, P.bloques, DIM_AE, max_epochs=AE_EPOCHS, seed=RANDOM_STATE + k)
    PROGRESO.paso(f"pliegue {k} | autoencoder", time.perf_counter() - t_ae)
    pesos_ae = reescalar_encoder(W, b, Z)

    # Severidad: sólo pólizas con siniestro
    tr = tr_pol[tr_pol.ClaimNb > 0].reset_index(drop=True)
    te = d[(d.pliegue == k) & (d.ClaimNb > 0)].reset_index(drop=True)
    log_tr, log_te = log_monto(tr), log_monto(te)
    pli_int = asignar_pliegues(tr, k=INNER_FOLDS, seed=RANDOM_STATE + k)
    X_tr_nn, X_te_nn = entradas_clases(P, tr), entradas_clases(P, te)

    resultado = {"pliegue": k, "n_train": len(tr), "n_test": len(te), "metodos": {}}
    oos = []

    for metodo in METODOS:
        cortes = FUNCIONES[metodo](log_tr)
        y_tr = etiquetar(log_tr, cortes)
        y_te = etiquetar(log_te, cortes)
        gvf_tr = gvf(log_tr, cortes)

        # GLM de referencia
        t_g = time.perf_counter()
        yhat_glm = glm_clases(P, tr, y_tr, te)
        PROGRESO.paso(f"pliegue {k} | {metodo} | GLM", time.perf_counter() - t_g)

        # FFNN: búsqueda con CV interna (cortes re-estimados en cada pliegue interno) y reentrenamiento final
        etiqueta = f"pliegue {k} | {metodo} | FFNN"
        cfg, f1_cv, _ = buscar_config_clases(metodo, tr, log_tr, P, pesos_ae, FUNCIONES, N_CONFIGS,
                                             pli_int, MAX_EPOCHS, rng, progreso=PROGRESO,
                                             etiqueta=etiqueta)
        yhat_nn = entrenar_final_clases(cfg, X_tr_nn, y_tr, X_te_nn, pesos_ae, MAX_EPOCHS,
                                        REPETICIONES, progreso=PROGRESO, etiqueta=etiqueta)

        for modelo, yhat in [("GLM", yhat_glm), ("FFNN", yhat_nn)]:
            resultado["metodos"][f"{metodo}|{modelo}"] = dict(
                **metricas_clasificacion(y_te, yhat), gvf_train=round(gvf_tr, 4),
                cortes_monto=np.round(np.expm1(cortes), 1).tolist())
        resultado["metodos"][f"{metodo}|FFNN"]["config"] = cfg
        resultado["metodos"][f"{metodo}|FFNN"]["cv_f1_macro"] = round(f1_cv, 4)

        oos.append(pd.DataFrame({"IDpol": te["IDpol"].to_numpy(), "metodo": metodo,
                                 "y": y_te, "yhat_GLM": yhat_glm, "yhat_FFNN": yhat_nn}))

    pd.concat(oos).to_csv(CARPETA / f"oos_pliegue{k}.csv.gz", index=False)
    resultado["segundos"] = round(time.time() - t0, 1)
    (CARPETA / f"pliegue_{k}.json").write_text(json.dumps(resultado, indent=2, default=str),
                                               encoding="utf-8")
    return resultado


for k in PLIEGUES_EXTERNOS:
    ruta = CARPETA / f"pliegue_{k}.json"
    if ruta.exists():
        print(f"Pliegue {k} ya existe, se omite")
        continue
    print(f"Pliegue externo {k} ...", flush=True)
    r = correr_pliegue(k)
    for clave, m in r["metodos"].items():
        print(f"  {clave:<28} F1={m['f1_macro']:.4f}  kappa={m['kappa_cuadratico']:.4f}  "
              f"bal_acc={m['balanced_accuracy']:.4f}")
    print(f"  tiempo {r['segundos']} s")

# %% [markdown]
# ## 3. Resultados fuera de muestra
#
# Métricas promediadas por pliegue. `gvf_train` mide qué tan bien separan los cortes la variabilidad de la severidad
# en el train. Los cortes del train se usan para etiquetar también el test.

# %%
filas = []
for ruta in sorted(CARPETA.glob("pliegue_*.json")):
    r = json.loads(ruta.read_text(encoding="utf-8"))
    for clave, m in r["metodos"].items():
        metodo, modelo = clave.split("|")
        filas.append({"pliegue": r["pliegue"], "metodo": metodo, "modelo": modelo,
                      "f1_macro": m["f1_macro"], "kappa_cuadratico": m["kappa_cuadratico"],
                      "balanced_accuracy": m["balanced_accuracy"], "mae_ordinal": m["mae_ordinal"],
                      "accuracy": m["accuracy"], "gvf_train": m["gvf_train"]})
res = pd.DataFrame(filas)
resumen = (res.groupby(["metodo", "modelo"])[["f1_macro", "kappa_cuadratico", "balanced_accuracy",
                                              "mae_ordinal", "accuracy", "gvf_train"]]
           .mean().round(4).reset_index())
try:
    display(resumen)
except NameError:          # ejecución como script fuera de Jupyter
    print(resumen.to_string(index=False))
resumen.to_csv(CARPETA / "resumen_clases.csv", index=False)
