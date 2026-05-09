#!/bin/bash

#SBATCH --job-name=blind-earth-qwen3-4b-reasoning-n20
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_qwen3_4b_reasoning_n20_%j.log
#SBATCH --error=logs/blind_earth_qwen3_4b_reasoning_n20_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Higher-fidelity reasoning run with n=20 votes per coordinate.
# Each pixel can now take 21 possible P(Land) values (0/20, ..., 20/20),
# which should make the map look much smoother than the n=5 (6 levels) and
# n=10 (11 levels) versions.
#
# Runtime expectation: roughly 2x the n=10 run (~24h based on n=10 = 12h),
# so the SLURM time limit is set to 30h with a healthy margin.
#
# Workers stay at 32 — each request now spawns 20 concurrent generations,
# so 32 x 20 = 640 in flight. KV cache may get tight; if vLLM logs show
# >90% utilization, drop workers to 24.
#
# Output: Generated models/Qwen_Qwen3-4B_reasoning_n20.png

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
NUM_SAMPLES=20

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
