#!/bin/bash

#SBATCH --job-name=blind-earth-diverse-v5
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_diverse_v5_%j.log
#SBATCH --error=logs/blind_earth_diverse_v5_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Fifth diverse-models batch: large models (>27B) in 4-bit AWQ form so they
# fit on a single 48 GB GPU. TP=1 (matching the reliable path from v4 and
# the bf16 → TP=2 hang seen in v3).
#
#   Qwen/Qwen2.5-72B-Instruct-AWQ        (72B - official Qwen AWQ, ~40 GB in mem)
#   casperhansen/mixtral-instruct-awq     (47B MoE - Mixtral-8x7B AWQ, ~28 GB)

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"
export VLLM_ENGINE_READY_TIMEOUT_S=7200

if [ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ]; then
    export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
fi

source .venv/bin/activate

MODELS=(
    "Qwen/Qwen2.5-72B-Instruct-AWQ"
    "casperhansen/mixtral-instruct-awq"
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

echo "Starting blind earth experiment (diverse models v5, large AWQ)..."
echo "Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "${MODELS[@]}" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

echo ""
echo "Experiment completed at $(date)"
