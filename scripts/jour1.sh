#!/usr/bin/env bash
# Jour 1 sur le GX10, d'une traite et relançable après un arrêt (chaque étape faite est sautée) : modèles, image,
# serveurs, corpus en profil large, juge (small contre large), échantillon d'hallucinations, rapport et archive.
#   scripts/jour1.sh             une vidéo à la fois
#   JOBS=2 scripts/jour1.sh      deux vidéos à la fois (débit ; chaque pipeline ajoute ~6 à 8 Gio)
# Avant, copier depuis la 5080 : corpus/media, out/corpus/commun (précalcul) et les JSON de out/corpus/small.
# Chronologie de la journée : out/jour1.log ; mémoire de l'hôte à 1 Hz : out/memoire.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p out/corpus
log() { echo "$(date -Iseconds) $*" | tee -a out/jour1.log; }

if [ ! -d corpus/media ] || [ ! -d out/corpus/commun ]; then
  echo "manque corpus/media ou out/corpus/commun : les copier depuis la 5080 (README, « Sur le GX10 »)" >&2; exit 1
fi

log "modèles (une fois, avec internet) et image du pipeline"
[ -f out/.modeles_gx10 ] || { scripts/docker.sh fetch gx10 && touch out/.modeles_gx10; }
docker image inspect audesia-worker >/dev/null 2>&1 || scripts/docker.sh build

log "serveurs : vision, puis rédacteur"
scripts/docker.sh start gx10

log "corpus en profil large, ${JOBS:-1} vidéo(s) à la fois"
python3 eval/run_corpus.py --profile large --docker --jobs "${JOBS:-1}" 2>&1 | tee -a out/corpus/run_large.log \
  || log "des vidéos ont échoué : relancer scripts/jour1.sh, qui reprend là où il s'est arrêté"
scripts/docker.sh stop

if ls out/corpus/small/*/ad.json >/dev/null 2>&1; then
  log "juge : small contre large, à l'aveugle"
  for d in out/corpus/small/*/; do  # le juge extrait ses images de l'extrait du premier run
    [ -f "$d/clip.mp4" ] || ln -f "out/corpus/commun/$(basename "$d")/clip.mp4" "$d/clip.mp4"
  done
  # Réglages du juge jamais essayés sur un GB10 : s'il ne démarre pas, la journée continue jusqu'à l'archive.
  if [ -f out/corpus/juge_small_vs_large.json ]; then
    log "juge déjà fait"
  elif { [ -f out/.modeles_juge ] || { scripts/docker.sh fetch juge && touch out/.modeles_juge; }; } \
      && scripts/docker.sh start juge; then
    scripts/docker.sh py eval/judge.py out/corpus/small out/corpus/large 2>&1 | tee -a out/corpus/juge_large.log \
      || log "le juge a échoué : relancer scripts/jour1.sh"
  else
    docker logs --tail 200 audesia-judge > out/juge_demarrage.log 2>&1 || true
    log "le juge n'a pas démarré : out/juge_demarrage.log et out/memoire.csv"
  fi
  scripts/docker.sh stop
  [ -f out/corpus/hallucinations_small_vs_large.json ] \
    || scripts/docker.sh py eval/hallucination_sample.py out/corpus/small out/corpus/large || log "échantillon en échec"
else
  log "pas de résultats small : juge et échantillon sautés"
fi

log "rapport et archive"
python3 eval/report.py >/dev/null
scripts/docker.sh py eval/archive.py large >/dev/null
log "fini : out/corpus/rapport.md, archive dans results/"
