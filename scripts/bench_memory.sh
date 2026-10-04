#!/usr/bin/env bash
# Relevé mémoire à 1 Hz sur l'hôte, et garde-fou. Sur GB10, nvidia-smi n'affiche pas la mémoire et docker stats
# ne voit pas les allocations CUDA : on lit /proc/meminfo, plus la mémoire par processus que donne nvidia-smi.
# Si la mémoire disponible passe sous FLOOR_GIB (8 Gio par défaut), les serveurs vLLM sont tués avant que la
# machine ne gèle.
#   scripts/bench_memory.sh [fichier.csv]     (lancé par scripts/docker.sh start gx10 et start juge)
set -u
out=${1:-out/memoire.csv}
floor_kib=$((${FLOOR_GIB:-8} * 1024 * 1024))
mkdir -p "$(dirname "$out")"
[ -s "$out" ] || echo "horodatage,utilisee_gib,disponible_gib,cache_gib,swap_gib,processus_gpu_mib" > "$out"
while true; do
  line=$(awk -v t="$(date -Iseconds)" '
    /^MemTotal:/ {tot = $2} /^MemAvailable:/ {av = $2} /^Cached:/ {ca = $2} /^SwapTotal:/ {st = $2} /^SwapFree:/ {sf = $2}
    END {g = 1048576; printf "%s,%.2f,%.2f,%.2f,%.2f,%d", t, (tot - av) / g, av / g, ca / g, (st - sf) / g, av}' /proc/meminfo)
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
  echo "${line%,*},\"$apps\"" >> "$out"
  if [ "${line##*,}" -lt "$floor_kib" ]; then
    echo "$(date -Iseconds) mémoire disponible sous ${FLOOR_GIB:-8} Gio : serveurs vLLM tués" >&2
    docker kill audesia-writer audesia-vision audesia-judge >/dev/null 2>&1
  fi
  sleep 1
done
