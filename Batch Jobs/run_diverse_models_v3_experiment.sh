#!/bin/bash

#SBATCH --job-name=blind-earth-diverse-v3
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_diverse_v3_%j.log
#SBATCH --error=logs/blind_earth_diverse_v3_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"

# vLLM reads this at Python import time, must be set in the shell before Python starts.
export VLLM_ENGINE_READY_TIMEOUT_S=7200

source .venv/bin/activate

# Models to run (all freely accessible, no token required, 8–14B):
#   deepseek-ai/DeepSeek-R1-Distill-Llama-8B   (8B  - DeepSeek R1 reasoning, Llama arch)
#   mistralai/Ministral-8B-Instruct-2410        (8B  - Mistral Ministral)
#   microsoft/Phi-3-medium-4k-instruct          (14B - Microsoft Phi-3 medium)
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "mistralai/Ministral-8B-Instruct-2410"
    "microsoft/Phi-3-medium-4k-instruct"
)

echo "Pre-downloading model weights to cache..."
for MODEL in "${MODELS[@]}"; do
    echo "  -> $MODEL"
    /ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "     [warning] Download failed for $MODEL"
done
echo "Pre-download complete."
echo ""

echo "Starting blind earth experiment (diverse models v3)..."
echo "Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "${MODELS[@]}" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2

echo ""
echo "Experiment completed at $(date)"
