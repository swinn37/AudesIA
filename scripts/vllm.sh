#!/usr/bin/env bash
# Serveurs vLLM des profils large (GX10) et test-vllm (RTX 5080), dans l'image officielle.
#   scripts/vllm.sh fetch gx10   télécharge les modèles du GX10, une seule fois, avec internet
#   scripts/vllm.sh start gx10   vision (port 8000) puis rédacteur (port 8001), l'un après l'autre
#   scripts/vllm.sh fetch 5080   télécharge Qwen3.5-2B (4,6 Go)
#   scripts/vllm.sh start 5080   un seul serveur (port 8000) pour tester le chemin vLLM sur 16 Go
#   scripts/vllm.sh stop         arrête et supprime les serveurs
# Sur le GX10, start lance aussi scripts/bench_memory.sh : relevé mémoire à 1 Hz et garde-fou.
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash sous Windows : ne pas réécrire les chemins du conteneur

# v0.30.0 plutôt que v0.29.0 : prompts avec images de Gemma 4 jusqu'à 3-4 fois plus rapides, préremplissage de
# Qwen3.6 corrigé sur GB10. Même base CUDA 13.0 ; les images NGC vLLM >= 26.04 refusent le pilote R580.
# Repli : vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1
IMG=${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90}
TEMPLATE=/root/.cache/huggingface/templates/gemma4  # modèle de conversation de Google (images en image_url)
COMMON=(--gpus all --ipc=host -v audesia-hf:/root/.cache/huggingface -v audesia-vllm:/root/.cache/vllm
        -e HF_HUB_OFFLINE="${OFFLINE:-1}" -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1)

hf() {  # téléchargement par le client Hugging Face de l'image vLLM : rien à installer sur l'hôte
  docker run --rm -v audesia-hf:/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" --entrypoint hf "$IMG" download "$@"
}

drop_caches() {  # GB10 : jusqu'à ~28 Gio de cache de pages à rendre avant chaque chargement
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "sans sudo : cache de pages non vidé" >&2
}

wait_ready() {  # $1 conteneur, $2 port : 30 min au plus, journal affiché si le serveur s'arrête
  for _ in $(seq 360); do
    curl -sf "http://127.0.0.1:$2/health" >/dev/null && { echo "$1 prêt"; return; }
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
    hf google/gemma-4-26B-A4B-it chat_template.jinja --local-dir "$TEMPLATE" ;;
  "fetch 5080")
    hf Qwen/Qwen3.5-2B ;;
  "start gx10")
    # Deux serveurs sur la mémoire unifiée : un démarrage trop gourmand a déjà gelé un GB10 (vLLM #46307), et à
    # distance, une machine gelée ne se redémarre pas. Donc 0,40 + 0,25 de la mémoire au plus, démarrages l'un
    # après l'autre, et garde-fou qui arrête les serveurs si la mémoire disponible passe sous 8 Gio.
    mkdir -p out
    nohup "$(dirname "$0")/bench_memory.sh" out/memoire.csv >> out/memoire.log 2>&1 &
    drop_caches
    # DeepGEMM dégrade la précision de Qwen3.6 sur Blackwell (vLLM #50332) : MoE en Triton.
    docker run -d --name audesia-vision "${COMMON[@]}" -e VLLM_USE_DEEP_GEMM=0 -p 127.0.0.1:8000:8000 "$IMG" \
      Qwen/Qwen3.6-35B-A3B-FP8 --host 0.0.0.0 --port 8000 \
      --gpu-memory-utilization 0.40 --max-model-len 65536 --max-num-seqs 8 --max-num-batched-tokens 8192 \
      --limit-mm-per-prompt '{"image": 16, "video": 0}' --mm-processor-cache-gb 0 --moe-backend triton \
      --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
    wait_ready audesia-vision 8000
    drop_caches
    docker run -d --name audesia-writer "${COMMON[@]}" -p 127.0.0.1:8001:8001 "$IMG" \
      nvidia/Gemma-4-26B-A4B-NVFP4 --host 0.0.0.0 --port 8001 \
      --gpu-memory-utilization 0.25 --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 \
      --limit-mm-per-prompt '{"image": 16, "audio": 0, "video": 0}' --mm-processor-kwargs '{"max_soft_tokens": 560}' \
      --mm-processor-cache-gb 0 --chat-template "$TEMPLATE/chat_template.jinja" \
      --reasoning-parser gemma4 --default-chat-template-kwargs '{"enable_thinking": false}'
    wait_ready audesia-writer 8001 ;;
  "start 5080")
    # 0,60 de 16 Go (~9,5 Gio) : Windows en occupe déjà ~3.
    docker run -d --name audesia-vision "${COMMON[@]}" -p 127.0.0.1:8000:8000 "$IMG" \
      Qwen/Qwen3.5-2B --host 0.0.0.0 --port 8000 \
      --gpu-memory-utilization 0.60 --max-model-len 32768 --max-num-seqs 4 \
      --limit-mm-per-prompt '{"image": 10, "video": 0}' --mm-processor-cache-gb 0 \
      --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
    wait_ready audesia-vision 8000 ;;
  "stop "*)
    docker rm -f audesia-writer audesia-vision 2>/dev/null || true
    pkill -f bench_memory.sh 2>/dev/null || true ;;
  *)
    sed -n '2,8p' "$0"; exit 2 ;;
esac
