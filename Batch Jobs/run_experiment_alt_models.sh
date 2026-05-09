#!/bin/bash

#SBATCH --job-name=blind-earth-alt-models
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_alt_%j.log
#SBATCH --error=logs/blind_earth_alt_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate venv
source .venv/bin/activate

# Run experiment with ALTERNATIVE lightweight open-source models
# Models: TinyLlama (1.1B), Phi-3-mini (3.8B), Mistral-7B (7B), Yi-6B (6B)
# All are publicly available without authentication
echo "Starting blind earth experiment (alternative lightweight models)..."
echo "Time: $(date)"
echo "Models: TinyLlama (1.1B), Phi-3-mini (3.8B), Mistral-7B (7B), Yi-6B (6B)"
echo "Resolution: 4 degrees"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --alt-models \
  --resolution 4 \
  --workers 32

echo ""
echo "Experiment completed at $(date)"
