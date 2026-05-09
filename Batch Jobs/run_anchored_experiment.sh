#!/bin/bash

#SBATCH --job-name=blind-earth-anchored
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_anchored_%j.log
#SBATCH --error=logs/blind_earth_anchored_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

mkdir -p logs

source .venv/bin/activate

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
export XDG_CACHE_HOME=/ceph/adinchev/.cache
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"

echo "Starting anchored experiment with $MODEL"
echo "Time: $(date)"
echo ""

# Run without anchor (baseline)
echo "=== Run 1/4: No anchor (baseline) ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --model "$MODEL" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

# Run with Mannheim anchor
echo "=== Run 2/4: Anchor = Mannheim ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --model "$MODEL" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --anchor mannheim \
  --reuse-existing-server

# Run with Tokyo anchor
echo "=== Run 3/4: Anchor = Tokyo ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --model "$MODEL" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --anchor tokyo \
  --reuse-existing-server

# Run with NYC anchor
echo "=== Run 4/4: Anchor = NYC ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --model "$MODEL" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --anchor nyc \
  --reuse-existing-server

echo ""
echo "All anchored experiments completed at $(date)"
