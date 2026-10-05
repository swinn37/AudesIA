# Image du pipeline (parole, plans, description, voix, mixage), pour le GX10 (arm64) et la RTX 5080 (amd64) :
# les dépendances seulement ; le code vient du dépôt, monté par scripts/docker.sh.
# Base : l'image vLLM des serveurs. Elle apporte CUDA 13, PyTorch et ffmpeg pour les deux architectures, démarre sur
# le pilote R580 du GX10, et ses couches y sont déjà téléchargées pour les serveurs.
#   docker build -t audesia-worker -f docker/Dockerfile .      (ou scripts/docker.sh build)
FROM vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90

# sox : sans lui, qwen-tts avertit à chaque lancement.
RUN apt-get update && apt-get install -y --no-install-recommends sox && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
# uv, comme l'image vLLM : il respecte ses surcharges de versions (UV_OVERRIDE), là où pip remplacerait des
# bibliothèques CUDA. Sans dépendances : qwen-tts tirerait gradio, qui ne sert qu'à sa démo ; scenedetect tirerait
# opencv-python, alors que l'image a déjà la version headless.
RUN uv pip install --system --no-cache -r /tmp/requirements.txt \
 && uv pip install --system --no-cache --no-deps qwen-tts==0.1.1 scenedetect==0.7.1

# 100 % local : aucun téléchargement ni télémétrie à l'exécution. Les modèles sont dans le cache monté, poids de
# Demucs (torchaudio) compris, grâce à TORCH_HOME.
ENV HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1 TORCH_HOME=/root/.cache/huggingface/torch
# Pas de copie du code : scripts/docker.sh monte le dépôt dans /work, et le code à jour tourne sans reconstruire
# l'image (une copie figée à la construction a déjà fait tourner un ancien pipeline sans prévenir).
WORKDIR /work
ENTRYPOINT ["python3"]
