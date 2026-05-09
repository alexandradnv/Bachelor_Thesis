#!/bin/bash

#SBATCH --job-name=blind-earth-olmo3-32b-ckpts
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_olmo3_32b_ckpts_%j.log
#SBATCH --error=logs/blind_earth_olmo3_32b_ckpts_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
export XDG_CACHE_HOME=/ceph/adinchev/.cache
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export HF_HUB_ENABLE_HF_TRANSFER=0

source .venv/bin/activate

echo "Starting blind earth experiment (OLMo3 32B checkpoint evolution)..."
echo "Time: $(date)"
echo "Models: 16 OLMo3-32B checkpoints (stage1 log-spaced + stage3 + main)"
echo "Resolution: 2 degrees"
echo "Tensor parallel: 2 GPUs"
echo "HF cache: $HF_HOME"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --test-olmo3-32b-evolution \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2

echo ""
echo "Experiment completed at $(date)"
