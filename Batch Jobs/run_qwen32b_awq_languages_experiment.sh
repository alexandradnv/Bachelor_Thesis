#!/bin/bash

#SBATCH --job-name=blind-earth-qwen32b-awq-languages
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_qwen32b_awq_languages_%j.log
#SBATCH --error=logs/blind_earth_qwen32b_awq_languages_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Language & cultural-bias experiment.
# Runs the same blind-earth probe in English / German / Spanish / Mandarin /
# Russian against Qwen2.5-32B-Instruct-AWQ. The vLLM server is started once
# and reused across all five language runs (--all-languages does this in a
# single process, so the model loads only once).

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_ENABLE_HF_TRANSFER=0

source .venv/bin/activate

MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"

echo "Starting Qwen2.5-32B-AWQ language-bias experiment..."
echo "Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --all-languages \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

echo ""
echo "All language runs completed at $(date)"
