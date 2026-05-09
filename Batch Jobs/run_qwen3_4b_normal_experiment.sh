#!/bin/bash

#SBATCH --job-name=blind-earth-qwen3-4b-normal
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_qwen3_4b_normal_%j.log
#SBATCH --error=logs/blind_earth_qwen3_4b_normal_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Qwen3-4B in NON-THINKING mode — runs the standard logprob-based blind-earth
# probe (one generated token per coord, no <think>...</think>). The
# --no-thinking flag passes chat_template_kwargs={enable_thinking: False} so
# the very first token is the final answer.
# Should run dramatically faster than the reasoning variant (~minutes, not
# half a day).
#
# Optimizations applied:
#   * --workers 48        (up from 16) - more in-flight chat completions
#   * vllm_server.py no longer passes --enforce-eager (CUDA graphs enabled)

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

echo "Pre-downloading $MODEL weights..."
/ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "[warning] Download failed"
echo ""

echo "Starting Qwen3-4B normal (non-thinking) experiment..."
echo "  Mode: chat completions, max_tokens=1, top_logprobs=20 (--no-thinking)"
echo "  Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --no-thinking \
  --resolution 2 \
  --workers 48 \
  --tensor-parallel-size 1

echo ""
echo "Qwen3-4B normal experiment completed at $(date)"
