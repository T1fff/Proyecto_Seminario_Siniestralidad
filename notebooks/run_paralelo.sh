#!/bin/bash
# Corre los pliegues externos en tres procesos en paralelo, 2 hilos de TensorFlow cada uno.
# Muestra el avance de cada grupo en esta misma terminal, cada 30 segundos.
# Uso (con el entorno de conda activo):
#   bash notebooks/run_paralelo.sh
cd "$(dirname "$0")" || exit 1
mkdir -p logs

GRUPOS=("1,2" "3,4" "5,6")
PIDS=()
NOMBRES=()
for g in "${GRUPOS[@]}"; do
  nombre=${g//,/_}
  PLIEGUES="$g" TF_THREADS=2 python 06_NeuralNetworks.py > "logs/pliegues_${nombre}.log" 2>&1 &
  PIDS+=($!)
  NOMBRES+=("$g")
done

mostrar_avance() {
  echo "--- $(date '+%H:%M:%S') ---"
  for i in "${!GRUPOS[@]}"; do
    nombre=${GRUPOS[$i]//,/_}
    linea=$(grep -E '^\s*\[ *[0-9]+/[0-9]+\]' "logs/pliegues_${nombre}.log" 2>/dev/null | tail -n 1)
    if [ -z "$linea" ]; then
      linea=$(tail -n 1 "logs/pliegues_${nombre}.log" 2>/dev/null)
    fi
    echo "  pliegues ${NOMBRES[$i]}: ${linea:-iniciando...}"
  done
}

vivos() {
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && return 0
  done
  return 1
}

while vivos; do
  mostrar_avance
  sleep 30
done

FALLOS=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FALLOS=$((FALLOS + 1))
done

mostrar_avance
if [ "$FALLOS" -eq 0 ]; then
  echo "Terminado sin errores. Resultados en notebooks/resultados/"
else
  echo "Terminaron $FALLOS proceso(s) con error. Revisa los logs en notebooks/logs/"
fi
