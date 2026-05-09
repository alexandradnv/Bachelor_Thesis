#!/bin/bash

#SBATCH --job-name=blind-earth-diverse-v6
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_diverse_v6_%j.log
#SBATCH --error=logs/blind_earth_diverse_v6_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Sixth diverse-models batch: mid-range (>7B, ≤32B), no gated models,
# AWQ for the >14B slot so it fits on a single 48 GB GPU. TP=1 everywhere.
#
#   Qwen/Qwen2.5-Coder-14B-Instruct           (14B bf16  - Qwen coder 14B)
#   Qwen/Qwen2.5-Coder-32B-Instruct-AWQ       (32B AWQ   - Qwen coder 32B, 4-bit)
#   upstage/SOLAR-10.7B-Instruct-v1.0         (10.7B bf16 - Upstage SOLAR, new family)
#   microsoft/Phi-3-medium-4k-instruct         (14B bf16  - Phi-3 medium, TP=1 retry)

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"
export VLLM_ENGINE_READY_TIMEOUT_S=7200

if [ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ]; then
    export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
fi

source .venv/bin/activate

MODELS=(
    "Qwen/Qwen2.5-Coder-14B-Instruct"
    "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
    "upstage/SOLAR-10.7B-Instruct-v1.0"
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

echo "Starting blind earth experiment (diverse models v6, mid-range)..."
echo "Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "${MODELS[@]}" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

echo ""
echo "Experiment completed at $(date)"
