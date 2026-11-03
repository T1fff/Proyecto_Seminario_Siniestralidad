# %%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Optimización para Apple Silicon (MacBook Pro M5): TensorFlow usa el GPU vía Metal si está
# instalado tensorflow-metal, y todos los núcleos de CPU para el resto. La estructura y los
# hiperparámetros no cambian.
import tensorflow as tf
_hilos = int(os.environ.get("TF_THREADS", "0"))   # 0 = todos; en paralelo, 2 por proceso
tf.config.threading.set_intra_op_parallelism_threads(_hilos)
tf.config.threading.set_inter_op_parallelism_threads(1 if _hilos else 0)
_gpus = tf.config.list_physical_devices("GPU")
print("Dispositivo:", "GPU (Metal) " + str(_gpus) if _gpus else "CPU (sin GPU Metal)")

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent))

from src.utils import load_freq, load_sev
from src.modelado import preparar_datos
from nn_holvoet import (
    preparar_fr, asignar_pliegues, Preprocesamiento, entrenar_autoencoder, reescalar_encoder,
    buscar_hiperparametros, entrenar_final, glm_predict, metricas, pesos_muestra, objetivo_de,
    entradas_de, RANDOM_STATE, Progreso,
)

# Protocolo. El paper usa N_CONFIGS=40, INNER_FOLDS=5, REPETICIONES=3, MAX_EPOCHS=500.
MODO_PRUEBA = os.environ.get("NN_PRUEBA") == "1"
if MODO_PRUEBA:
    CARPETA = HERE / "resultados_prueba"
    N_CONFIGS, INNER_FOLDS, REPETICIONES, MAX_EPOCHS, AE_EPOCHS = 2, 2, 1, 5, 5
    PLIEGUES_EXTERNOS = [1]
else:
    CARPETA = HERE / "resultados"
    N_CONFIGS, INNER_FOLDS, REPETICIONES, MAX_EPOCHS, AE_EPOCHS = 4, 3, 1, 50, 50
    PLIEGUES_EXTERNOS = [int(x) for x in os.environ.get("PLIEGUES", "1,2,3,4,5,6").split(",")]
N_PLIEGUES = 6
DIM_AE = 15            # FR15 en el código de los autores (NNforFS_4, línea 145)
CANN_FIJA = True       # CANN fija: w_NN = w_IN = 1, b = 0 (Ec. 4)
MODELOS_NN = ["FFNN", "CANN"]
CARPETA.mkdir(parents=True, exist_ok=True)

# Unidades de trabajo para el progreso: autoencoder, GLM (freq y sev), y CV interna + repeticiones finales por modelo y tarea
UNIDADES_POR_PLIEGUE = 1 + 2 + 2 * 2 * (N_CONFIGS * INNER_FOLDS + REPETICIONES)   # AE + 2 GLM + NN
PROGRESO = Progreso(total=len(PLIEGUES_EXTERNOS) * UNIDADES_POR_PLIEGUE)
print(f"Unidades de entrenamiento: {PROGRESO.total}")

# %%
df = preparar_datos(load_freq("freMTPL2freq.csv"), load_sev("freMTPL2sev.csv"))
d = preparar_fr(df)
d["pliegue"] = asignar_pliegues(d, k=N_PLIEGUES, seed=RANDOM_STATE)
print(f"Pólizas: {len(d):,} | con siniestro: {(d.ClaimNb > 0).sum():,}")
print(d.groupby("pliegue").size().rename("pólizas").to_frame().T.to_string())

# %%
import subprocess
from IPython.display import clear_output

PARALELO = True            # False: no lanza procesos; la celda siguiente corre en secuencia
GRUPOS_PARALELOS = ["1,2", "3,4", "5,6"]

# Guarda: el propio script, al ejecutarse en paralelo, tiene la variable PLIEGUES y no vuelve a lanzar procesos.
if PARALELO and "PLIEGUES" not in os.environ:
    CARPETA_LOGS = HERE / "logs"
    CARPETA_LOGS.mkdir(exist_ok=True)
    procesos = []
    for g in GRUPOS_PARALELOS:
        log = CARPETA_LOGS / f"pliegues_{g.replace(',', '_')}.log"
        env = {**os.environ, "PLIEGUES": g, "TF_THREADS": "2"}
        p = subprocess.Popen([sys.executable, str(HERE / "06_NeuralNetworks.py")],
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
        print(f"pliegues {g}: código de salida {p.returncode} (log: logs/{log.name})")


# %%
def correr_pliegue(k):
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_STATE + k)
    tr = d[d.pliegue != k].reset_index(drop=True)
    te = d[d.pliegue == k].reset_index(drop=True)
    P = Preprocesamiento().fit(tr)

    # Autoencoder (no supervisado, con datos de frecuencia del train)
    Z = P.categoricas(tr)
    t_ae = time.perf_counter()
    W, b = entrenar_autoencoder(Z, P.bloques, DIM_AE, max_epochs=AE_EPOCHS, seed=RANDOM_STATE + k)
    PROGRESO.paso(f"pliegue {k} | autoencoder", time.perf_counter() - t_ae)
    pesos_ae = reescalar_encoder(W, b, Z)

    # Pliegues internos para la búsqueda (estratificados por siniestro)
    pli_int = asignar_pliegues(tr, k=INNER_FOLDS, seed=RANDOM_STATE + k)

    resultado = {"pliegue": k, "modelos": {}}

    for tarea in ["freq", "sev"]:
        if tarea == "freq":
            tr_t, te_t = tr, te
        else:
            tr_t = tr[tr.ClaimNb > 0].reset_index(drop=True)
            te_t = te[te.ClaimNb > 0].reset_index(drop=True)
            pli_t = pli_int[(tr.ClaimNb > 0).to_numpy()]
        pli_use = pli_int if tarea == "freq" else pli_t

        # GLM de referencia y ŷ_IN para la CANN (entrenado sólo con el train de este pliegue)
        t_glm = time.perf_counter()
        glm_tr, glm_te = glm_predict(tr, te, tarea, P)
        PROGRESO.paso(f"pliegue {k} | GLM {tarea}", time.perf_counter() - t_glm)
        if tarea == "sev":
            glm_tr_s = glm_tr[(tr.ClaimNb > 0).to_numpy()]
            glm_te_s = glm_te[(te.ClaimNb > 0).to_numpy()]
        else:
            glm_tr_s, glm_te_s = glm_tr, glm_te

        y_te = objetivo_de(te_t, tarea)
        w_te = pesos_muestra(te_t, tarea)
        mu_glm = glm_te_s
        resultado["modelos"][f"GLM|{tarea}"] = metricas(tarea, y_te, mu_glm, w_te)
        out_t = (te_t[["IDpol"]].assign(modelo="GLM", tarea=tarea, pred=mu_glm,
                                        y=y_te, w=w_te))

        for modelo in MODELOS_NN:
            cann = modelo == "CANN"
            t1 = time.time()
            etiqueta = f"pliegue {k} | {modelo} {tarea}"
            cfg, dev_cv, _ = buscar_hiperparametros(
                tarea, tr_t, P, pesos_ae, cann, glm_tr_s if cann else None,
                N_CONFIGS, pli_use, MAX_EPOCHS, rng, log=lambda *a: None,
                progreso=PROGRESO, etiqueta=etiqueta)
            mu, n_rep = entrenar_final(
                tarea, cfg, tr_t, P, pesos_ae, cann, glm_tr_s if cann else None,
                glm_te_s if cann else None, te_t, REPETICIONES, MAX_EPOCHS, cann_fija=CANN_FIJA,
                progreso=PROGRESO, etiqueta=etiqueta)
            resultado["modelos"][f"{modelo}|{tarea}"] = dict(
                **metricas(tarea, y_te, mu, w_te),
                cv_deviance=dev_cv, config=cfg, repeticiones=n_rep,
                segundos=round(time.time() - t1, 1))
            out_t = pd.concat([out_t, te_t[["IDpol"]].assign(
                modelo=modelo, tarea=tarea, pred=mu, y=y_te, w=w_te)])

        out_t.to_csv(CARPETA / f"oos_pliegue{k}_{tarea}.csv.gz", index=False)

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
    for clave, m in r["modelos"].items():
        print(f"  {clave:<14} deviance={m['deviance']:.5f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")
    print(f"  tiempo {r['segundos']} s")

# %%
filas = []
for ruta in sorted(CARPETA.glob("pliegue_*.json")):
    r = json.loads(ruta.read_text(encoding="utf-8"))
    for clave, m in r["modelos"].items():
        modelo, tarea = clave.split("|")
        filas.append({"pliegue": r["pliegue"], "tarea": tarea, "modelo": modelo,
                      **{q: m[q] for q in ["deviance", "mae", "rmse"]}})
res = pd.DataFrame(filas)
resumen = (res.groupby(["tarea", "modelo"])[["deviance", "mae", "rmse"]]
           .mean().round(5).reset_index())
try:
    display(resumen)
except NameError:          # ejecución como script fuera de Jupyter
    print(resumen.to_string(index=False))
resumen.to_csv(CARPETA / "resumen_metricas.csv", index=False)
