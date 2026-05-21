#!/bin/bash

#SBATCH --job-name=blind-earth-gemma3-12b-normal
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_gemma3_12b_normal_%j.log
#SBATCH --error=logs/blind_earth_gemma3_12b_normal_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Standard blind-earth land/water probe for google/gemma-3-12b-it.
# No --language flag → uses the original English "Land"/"Water" prompt
# (see language_runs default in blind_model_experiment.py).
# Fits on a single 48 GB GPU in bf16.

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

MODEL="google/gemma-3-12b-it"

echo "Pre-downloading $MODEL weights..."
/ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "[warning] Download failed — model may require HF token with accepted Gemma license"
echo ""

# Pick a unique port derived from the SLURM job id so we never collide with a
# stale vLLM left behind by another user on the same shared node. (Job
# 245846 landed on dws-10 where another user's vLLM was already bound to
# 8000 serving Qwen2.5-3B; our pkill couldn't kill it and all queries 404'd.)
VLLM_PORT=$((30000 + SLURM_JOB_ID % 30000))
BASE_URL="http://localhost:${VLLM_PORT}/v1"
echo "Using vLLM port: $VLLM_PORT"

echo "Starting Gemma3-12B normal (English land/water) experiment..."
echo "Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --base-url "$BASE_URL"

echo ""
echo "Gemma3-12B normal experiment completed at $(date)"
