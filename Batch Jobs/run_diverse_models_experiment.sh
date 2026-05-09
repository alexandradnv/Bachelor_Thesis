#!/bin/bash

#SBATCH --job-name=blind-earth-diverse-models
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_diverse_%j.log
#SBATCH --error=logs/blind_earth_diverse_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Create logs directory if it doesn't exist
mkdir -p logs

# Redirect HuggingFace cache to /ceph to avoid home directory quota limits
export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"

# Activate venv
source .venv/bin/activate

# Run experiment with diverse open-source models (7B–32B) from different families:
#   mistralai/Mistral-Nemo-Instruct-2407     (12B)
#   mistralai/Mistral-Small-24B-Instruct-2501 (24B)
#   microsoft/phi-4                          (14B)
#   Qwen/Qwen2.5-32B-Instruct               (32B)
#   allenai/OLMo-3-1025-7B-Instruct         (7B)
#   tiiuae/Falcon3-10B-Instruct             (10B)
#   ibm-granite/granite-3.1-8b-instruct     (8B)
echo "Starting blind earth experiment (diverse open-source models, 7B–32B)..."
echo "Time: $(date)"
echo "Resolution: 2 degrees"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --diverse-models \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2

echo ""
echo "Experiment completed at $(date)"
