#!/usr/bin/env bash
# Audesia sous Docker : serveurs vLLM et pipeline sur un réseau sans accès sortant, preuve du 100 % local.
#   scripts/docker.sh fetch gx10     modèles du GX10, une seule fois, avec internet (environ 64 Go)
#   scripts/docker.sh fetch 5080     Qwen3.5-2B (4,6 Go), pour tester sur la RTX 5080
#   scripts/docker.sh build          image du pipeline (audesia-worker), construite sur la machine qui l'utilise
#   scripts/docker.sh start gx10     vision (Qwen3.6) puis rédacteur (Gemma 4), l'un après l'autre
#   scripts/docker.sh start 5080     un seul serveur (Qwen3.5-2B) qui joue les deux rôles
#   scripts/docker.sh run ARGS...    pipeline, par exemple : run film.mkv --start 1:35 --end 3:35 --profile large
#   scripts/docker.sh py SCRIPT ...  un script Python du dépôt dans la même image, par exemple eval/overlap.py
#   scripts/docker.sh stop           arrête les serveurs
# À lancer depuis la racine du dépôt : run monte ce dossier, les vidéos et out/ s'y trouvent.
# Sur le GX10, start lance aussi scripts/bench_memory.sh : relevé mémoire à 1 Hz et garde-fou.
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash sous Windows : ne pas réécrire les chemins du conteneur

# v0.30.0 plutôt que v0.29.0 : prompts avec images de Gemma 4 jusqu'à 3-4 fois plus rapides, préremplissage de
# Qwen3.6 corrigé sur GB10. Même base CUDA 13.0 ; les images NGC vLLM >= 26.04 refusent le pilote R580.
# Repli : vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1
# L'image du pipeline (docker/Dockerfile) part de la même image : ses couches sont partagées.
IMG=${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90}
NET=audesia-net                    # réseau Docker interne : aucun conteneur n'y a d'accès sortant
HF_CACHE=${HF_CACHE:-audesia-hf}   # cache des modèles du pipeline ; sur la 5080, celui de Windows évite de retélécharger
TEMPLATE=/root/.cache/huggingface/templates/gemma4  # modèle de conversation de Google (images en image_url)
SERVER=(--gpus all --ipc=host --network "$NET" -v audesia-hf:/root/.cache/huggingface -v audesia-vllm:/root/.cache/vllm
        -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1)
WORKER=(--rm --gpus all --ipc=host --network "$NET" -v "$HF_CACHE:/root/.cache/huggingface"
        -v "$(pwd -W 2>/dev/null || pwd):/work")  # le dépôt monté : vidéos, corpus et out/ aux mêmes chemins qu'en natif

hf() {  # téléchargement par le client Hugging Face de l'image vLLM : rien à installer sur l'hôte
  docker run --rm -v audesia-hf:/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" --entrypoint hf "$IMG" download "$@"
}

network() {
  docker network inspect "$NET" >/dev/null 2>&1 || docker network create --internal "$NET" >/dev/null
}

drop_caches() {  # GB10 : jusqu'à ~28 Gio de cache de pages à rendre avant chaque chargement
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "sans sudo : cache de pages non vidé" >&2
}

wait_ready() {  # $1 conteneur, $2 port. Santé lue depuis le conteneur : le réseau interne n'est pas joignable de l'hôte.
  for _ in $(seq 360); do  # 30 min au plus
    docker exec "$1" curl -sf "http://127.0.0.1:$2/health" >/dev/null 2>&1 && { echo "$1 prêt"; return; }
    if [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" != true ]; then
      docker logs --tail 60 "$1"; echo "$1 s'est arrêté" >&2; exit 1
    fi
    sleep 5
  done
  echo "$1 : pas prêt après 30 min" >&2; exit 1
}

case "${1:-} ${2:-}" in
  "fetch gx10")
    hf Qwen/Qwen3.6-35B-A3B-FP8
    hf nvidia/Gemma-4-26B-A4B-NVFP4
    # Le modèle NVFP4 livre un ancien modèle de conversation, qui ne reconnaît pas les images en image_url.
    hf google/gemma-4-26B-A4B-it chat_template.jinja --local-dir "$TEMPLATE"
    hf openai/whisper-large-v3 --include "*.json" "*.txt" "model.safetensors"  # pas les copies .bin et flax
    hf Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    # Poids Hybrid Demucs (335 Mo) pour les cris et souffles, dans le cache que monte le pipeline.
    docker run --rm -v audesia-hf:/root/.cache/huggingface -e TORCH_HOME=/root/.cache/huggingface/torch \
      --entrypoint python3 "$IMG" -c "import torchaudio; torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.get_model()" ;;
  "fetch 5080")
    hf Qwen/Qwen3.5-2B ;;
  "build "*)
    docker build -t audesia-worker -f docker/Dockerfile . ;;
  "start gx10")
    # Deux serveurs sur la mémoire unifiée : un démarrage trop gourmand a déjà gelé un GB10 (vLLM #46307), et à
    # distance, une machine gelée ne se redémarre pas. Donc 0,40 + 0,25 de la mémoire au plus, démarrages l'un
    # après l'autre, et garde-fou qui arrête les serveurs si la mémoire disponible passe sous 8 Gio.
    network
    mkdir -p out
    nohup "$(dirname "$0")/bench_memory.sh" out/memoire.csv >> out/memoire.log 2>&1 &
    drop_caches
    # DeepGEMM dégrade la précision de Qwen3.6 sur Blackwell (vLLM #50332) : MoE en Triton.
    docker run -d --name audesia-vision "${SERVER[@]}" -e VLLM_USE_DEEP_GEMM=0 "$IMG" \
      Qwen/Qwen3.6-35B-A3B-FP8 --host 0.0.0.0 --port 8000 \
      --gpu-memory-utilization 0.40 --max-model-len 65536 --max-num-seqs 8 --max-num-batched-tokens 8192 \
      --limit-mm-per-prompt '{"image": 16, "video": 0}' --mm-processor-cache-gb 0 --moe-backend triton \
      --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
    wait_ready audesia-vision 8000
    drop_caches
    docker run -d --name audesia-writer "${SERVER[@]}" "$IMG" \
      nvidia/Gemma-4-26B-A4B-NVFP4 --host 0.0.0.0 --port 8001 \
      --gpu-memory-utilization 0.25 --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 \
      --limit-mm-per-prompt '{"image": 16, "audio": 0, "video": 0}' --mm-processor-kwargs '{"max_soft_tokens": 560}' \
      --mm-processor-cache-gb 0 --chat-template "$TEMPLATE/chat_template.jinja" \
      --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking": false}'
    wait_ready audesia-writer 8001 ;;
  "start 5080")
    # 0,45 de 16 Go (~7 Gio) : Windows en occupe ~3, et le pipeline doit encore y charger la transcription puis la voix.
    network
    docker run -d --name audesia-vision "${SERVER[@]}" "$IMG" \
      Qwen/Qwen3.5-2B --host 0.0.0.0 --port 8000 \
      --gpu-memory-utilization 0.45 --max-model-len 32768 --max-num-seqs 4 \
      --limit-mm-per-prompt '{"image": 10, "video": 0}' --mm-processor-cache-gb 0 \
      --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
    wait_ready audesia-vision 8000 ;;
  "run "*)
    shift
    network
    writer=http://audesia-writer:8001/v1
    docker inspect audesia-writer >/dev/null 2>&1 || writer=http://audesia-vision:8000/v1  # 5080 : un seul serveur
    docker run "${WORKER[@]}" -e AUDESIA_VLM_URL=http://audesia-vision:8000/v1 -e AUDESIA_WRITER_URL="$writer" \
      audesia-worker "$@" ;;
  "py "*)  # un script Python du dépôt dans l'image du pipeline, par exemple : py eval/overlap.py ...
    shift
    network
    docker run "${WORKER[@]}" --entrypoint python3 audesia-worker "$@" ;;
  "stop "*)
    docker rm -f audesia-writer audesia-vision 2>/dev/null || true
    pkill -f bench_memory.sh 2>/dev/null || true ;;
  *)
    sed -n '2,12p' "$0"; exit 2 ;;
esac
