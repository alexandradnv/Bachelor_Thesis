#!/bin/bash

#SBATCH --job-name=blind-earth-qwen3-4b-reasoning
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_qwen3_4b_reasoning_%j.log
#SBATCH --error=logs/blind_earth_qwen3_4b_reasoning_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Reasoning-model variant of the blind-earth probe.
# Qwen3-4B is a thinking model: it emits <think>...</think> blocks before
# the final answer, so the max_tokens=1 + logprobs trick used elsewhere in
# this codebase doesn't work. Instead we:
#   1. Configure vLLM with --reasoning-parser qwen3 so the server splits
#      <think> from the final answer (final answer ends up in
#      choice.message.content, thinking ends up in reasoning_content).
#   2. Ask the model to wrap its verdict in <answer>Land</answer> /
#      <answer>Water</answer> and use stop=["</answer>"] in the sampling
#      params, so generation terminates the moment the model commits even
#      if it would have kept thinking. This is what fixes the previous run
#      where MSE was a flat 0.25 (every coord defaulted to 0.5 because the
#      model ran out of max_tokens mid-think and never emitted an answer).
#   3. For each coordinate, sample n=NUM_SAMPLES completions at temperature=0.7
#      and compute P(Land) = land_count / (land_count + water_count).
# For faster debugging swap MODEL to "Qwen/Qwen3-0.6B".

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
NUM_SAMPLES=5

echo "Pre-downloading $MODEL weights..."
/ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "[warning] Download failed"
echo ""

echo "Starting Qwen3-4B reasoning experiment..."
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
  --workers 48 \
  --tensor-parallel-size 1

echo ""
echo "Qwen3-4B reasoning experiment completed at $(date)"
