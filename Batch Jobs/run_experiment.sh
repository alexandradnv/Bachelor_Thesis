#!/bin/bash

#SBATCH --job-name=blind-earth-all-models
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_%j.log
#SBATCH --error=logs/blind_earth_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate venv
source .venv/bin/activate

# Run experiment with student-friendly models (publicly available, no authentication needed)
# Models: 0.5B, 1.5B, 3B, 7B (all Qwen, all publicly available)
# Note: LLaMA models are also gated - they require HuggingFace authentication
echo "Starting blind earth experiment (student-friendly models - publicly available)..."
echo "Time: $(date)"
echo "Models: 0.5B, 1.5B, 3B, 7B (Qwen only - no authentication needed)"
echo "Resolution: 4 degrees"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --student-models \
  --resolution 4 \
  --workers 32

echo ""
echo "Experiment completed at $(date)"
