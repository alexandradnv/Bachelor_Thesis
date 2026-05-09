#!/bin/bash

#SBATCH --job-name=blind-earth-qwen32b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_qwen32b_%j.log
#SBATCH --error=logs/blind_earth_qwen32b_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

mkdir -p logs

source .venv/bin/activate

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
export XDG_CACHE_HOME=/ceph/adinchev/.cache
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

echo "Starting blind earth experiment (Qwen2.5-32B-Instruct)..."
echo "Time: $(date)"
echo "GPUs: 2 (tensor parallelism)"
echo "Resolution: 2 degrees"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --model "Qwen/Qwen2.5-32B-Instruct-AWQ" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

echo ""
echo "Experiment completed at $(date)"
