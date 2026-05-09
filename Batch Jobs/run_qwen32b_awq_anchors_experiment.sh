#!/bin/bash

#SBATCH --job-name=blind-earth-qwen32b-awq-anchors
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_qwen32b_awq_anchors_%j.log
#SBATCH --error=logs/blind_earth_qwen32b_awq_anchors_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_ENABLE_HF_TRANSFER=0

source .venv/bin/activate

MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"

# Weights are already cached (19 GB), no download needed.

echo "Starting Qwen2.5-32B-AWQ multi-anchor experiment..."
echo "Time: $(date)"
echo ""

# Run 1/4: 5 anchors — starts the vLLM server (single GPU, AWQ is quantized)
echo "=== Run 1/4: 5 anchor cities ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --anchor-count 5 \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

# Runs 2–4: reuse the running server
echo "=== Run 2/4: 10 anchor cities ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --anchor-count 10 \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --reuse-existing-server

echo "=== Run 3/4: 15 anchor cities ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --anchor-count 15 \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --reuse-existing-server

echo "=== Run 4/4: 20 anchor cities ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --anchor-count 20 \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --reuse-existing-server

echo ""
echo "All AWQ anchor experiments completed at $(date)"
