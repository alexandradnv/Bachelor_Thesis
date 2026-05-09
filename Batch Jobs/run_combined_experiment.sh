#!/bin/bash

#SBATCH --job-name=blind-earth-combined
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_combined_%j.log
#SBATCH --error=logs/blind_earth_combined_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate venv
source .venv/bin/activate

export VLLM_ENGINE_READY_TIMEOUT_S=3600

# ---- FIX: redirect HuggingFace cache to /ceph to avoid /home quota issues ----
export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
export XDG_CACHE_HOME=/ceph/adinchev/.cache
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

# Disable hf_transfer (xet_get) which caused "Background writer channel closed" errors
export HF_HUB_ENABLE_HF_TRANSFER=0

# Run experiment with student models + OLMo3 checkpoints
echo "Starting blind earth experiment (Combined: Student Models + OLMo3 Checkpoints)..."
echo "Time: $(date)"
echo "Models: 4 student models + 14 OLMo3 checkpoints = 18 total"
echo "Resolution: 2 degrees (high quality)"
echo "HF cache: $HF_HOME"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --student-models \
  --test-olmo3-evolution \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

echo ""
echo "Experiment completed at $(date)"
