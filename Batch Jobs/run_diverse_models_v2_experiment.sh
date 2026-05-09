#!/bin/bash

#SBATCH --job-name=blind-earth-diverse-v2
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_diverse_v2_%j.log
#SBATCH --error=logs/blind_earth_diverse_v2_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Create logs directory if it doesn't exist
mkdir -p logs

# Redirect HuggingFace cache to /ceph to avoid home directory quota limits
export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"

# vLLM reads VLLM_ENGINE_READY_TIMEOUT_S at Python import time (not from subprocess env),
# so it must be set here in the shell before Python starts.
export VLLM_ENGINE_READY_TIMEOUT_S=7200

# Activate venv
source .venv/bin/activate

# Pre-download all model weights to the HF cache before starting vLLM.
# Without this, vLLM downloads ~65 GB weights at startup and hits its engine timeout.
echo "Pre-downloading model weights to cache (this may take a while for large models)..."
for MODEL in \
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
    "allenai/Olmo-3-7B-Think" \
    "allenai/Olmo-3-7B-Instruct"; do
    echo "  -> $MODEL"
    /ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "     [warning] Download failed for $MODEL"
done
echo "Pre-download step complete."
echo ""

# Run experiment with open-source reasoning models (no API keys or gated access required):
#   deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   (14B - reasoning)
#   allenai/Olmo-3-7B-Think                    (7B  - OLMo-3 Think, reasoning)
echo "Starting blind earth experiment (diverse models v2: reasoning models)..."
echo "Time: $(date)"
echo "Resolution: 2 degrees"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --diverse-models-v2 \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2

echo ""
echo "Experiment completed at $(date)"
