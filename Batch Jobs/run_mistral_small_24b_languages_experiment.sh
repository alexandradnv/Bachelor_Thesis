#!/bin/bash

#SBATCH --job-name=blind-earth-mistral-small-24b-languages
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_mistral_small_24b_languages_%j.log
#SBATCH --error=logs/blind_earth_mistral_small_24b_languages_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Language & cultural-bias experiment for mistralai/Mistral-Small-24B-Instruct-2501.
# Picked as a Gemma3-27B substitute because it's ungated (Apache 2.0 — no
# HF license to accept), 24B (close to 27B), and Mistral-Small is multilingual
# (Thai/Bulgarian/Russian/German all in its training mix).
# Needs 2x48 GB GPUs at bf16 (~48 GB just for weights). Runs all five
# languages reusing a single loaded server.

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export HF_HUB_ENABLE_HF_TRANSFER=0

source .venv/bin/activate

MODEL="mistralai/Mistral-Small-24B-Instruct-2501"

echo "Pre-downloading $MODEL weights..."
/ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "[warning] Download failed"
echo ""

echo "Starting Mistral-Small-24B language-bias experiment..."
echo "Languages: Russian, Bulgarian, English, German, Thai"
echo "Time: $(date)"
echo ""

# Run 1: Russian (cold-start: load the model)
echo "=== Run 1/5: Russian prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language ru \
  --resolution 2 \
  --workers 48 \
  --tensor-parallel-size 2

# Runs 2-5 reuse the already-loaded server so the model only loads once.
echo "=== Run 2/5: Bulgarian prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language bg \
  --resolution 2 \
  --workers 48 \
  --tensor-parallel-size 2 \
  --reuse-existing-server

echo "=== Run 3/5: English prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language en \
  --resolution 2 \
  --workers 48 \
  --tensor-parallel-size 2 \
  --reuse-existing-server

echo "=== Run 4/5: German prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language de \
  --resolution 2 \
  --workers 48 \
  --tensor-parallel-size 2 \
  --reuse-existing-server

echo "=== Run 5/5: Thai prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language th \
  --resolution 2 \
  --workers 48 \
  --tensor-parallel-size 2 \
  --reuse-existing-server

echo ""
echo "All Mistral-Small-24B language runs completed at $(date)"
