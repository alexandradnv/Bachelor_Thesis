#!/bin/bash

#SBATCH --job-name=blind-earth-gemma3-27b-languages
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_gemma3_27b_languages_%j.log
#SBATCH --error=logs/blind_earth_gemma3_27b_languages_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Language & cultural-bias experiment for google/gemma-3-27b-it.
# Needs 2x48 GB GPUs (tensor-parallel-size=2) — bf16 27B weights are ~54 GB
# and don't fit on a single 48 GB card. SLURM may queue this until the user's
# 2-GPU QOS quota frees up.
# Runs the blind-earth probe in Russian, Bulgarian, English, German, and Thai.
# Each language uses its own native words for "land" and "water" (configured
# in LANGUAGE_CONFIGS in blind_model_experiment.py). The vLLM server is
# loaded once and reused across all five language runs.

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export HF_HUB_ENABLE_HF_TRANSFER=0

# Gemma-3 is gated on Hugging Face — needs an HF token with the licence accepted.
if [ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ]; then
    export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
fi

source .venv/bin/activate

MODEL="google/gemma-3-27b-it"

echo "Pre-downloading $MODEL weights..."
/ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "[warning] Download failed — model may require HF token with accepted Gemma license"
echo ""

echo "Starting Gemma3-27B language-bias experiment..."
echo "Languages: Russian, Bulgarian, English, German, Thai"
echo "Time: $(date)"
echo ""

# Run 1: Russian (cold-start: load the model)
echo "=== Run 1/5: Russian prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language ru \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2

# Runs 2-5 reuse the already-loaded server so the model only loads once.
echo "=== Run 2/5: Bulgarian prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language bg \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2 \
  --reuse-existing-server

echo "=== Run 3/5: English prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language en \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2 \
  --reuse-existing-server

echo "=== Run 4/5: German prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language de \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2 \
  --reuse-existing-server

echo "=== Run 5/5: Thai prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language th \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 2 \
  --reuse-existing-server

echo ""
echo "All Gemma3-27B language runs completed at $(date)"
