#!/bin/bash

#SBATCH --job-name=blind-earth-qwen25-coder-32b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_qwen25_coder_32b_%j.log
#SBATCH --error=logs/blind_earth_qwen25_coder_32b_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Standard blind-earth land/water probe for Qwen/Qwen2.5-Coder-32B-Instruct (bf16).
# 2x48 GB GPUs (tensor-parallel-size=2) — bf16 32B weights are ~64 GB.

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export HF_HUB_ENABLE_HF_TRANSFER=0

source .venv/bin/activate

MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"

echo "Pre-downloading $MODEL weights..."
/ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "[warning] Download failed"
echo ""

VLLM_PORT=$((30000 + SLURM_JOB_ID % 30000))
BASE_URL="http://localhost:${VLLM_PORT}/v1"
echo "Using vLLM port: $VLLM_PORT"

echo "Starting Qwen2.5-Coder-32B-Instruct normal experiment..."
echo "Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2 \
  --base-url "$BASE_URL"

echo ""
echo "Experiment completed at $(date)"
