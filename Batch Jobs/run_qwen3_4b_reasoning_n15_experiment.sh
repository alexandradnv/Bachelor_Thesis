#!/bin/bash

#SBATCH --job-name=blind-earth-qwen3-4b-reasoning-n15
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=22:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_qwen3_4b_reasoning_n15_%j.log
#SBATCH --error=logs/blind_earth_qwen3_4b_reasoning_n15_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Reasoning run with n=15 votes per coordinate.
# Each pixel can take 16 possible P(Land) values (0/15, ..., 15/15),
# halfway between the n=10 (11 levels) and n=20 (21 levels) sweep.
#
# Runtime: ~18h expected (linear scaling: n=10 = 12h, so n=15 ~= 18h);
# SLURM time set to 22h.
#
# Output: Generated models/Qwen_Qwen3-4B_reasoning_n15.png

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export HF_HUB_ENABLE_HF_TRANSFER=0

if [ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ]; then
    export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
fi

source .venv/bin/activate

MODEL="Qwen/Qwen3-4B"
NUM_SAMPLES=15

echo "Pre-downloading $MODEL weights..."
/ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "[warning] Download failed"
echo ""

echo "Starting Qwen3-4B reasoning experiment (n=$NUM_SAMPLES)..."
echo "  Sampling: n=$NUM_SAMPLES per coordinate (majority vote -> P(Land))"
echo "  Reasoning parser: qwen3 (strips <think>...</think> server-side)"
echo "  Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --reasoning \
  --num-samples "$NUM_SAMPLES" \
  --reasoning-parser qwen3 \
  --max-model-len 2048 \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

echo ""
echo "Qwen3-4B reasoning experiment (n=$NUM_SAMPLES) completed at $(date)"
