#!/bin/bash

#SBATCH --job-name=blind-earth-bggpt-27b-bg
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_bggpt_27b_bg_%j.log
#SBATCH --error=logs/blind_earth_bggpt_27b_bg_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Bulgarian language-bias probe — bigger-model variant.
# Runs the blind-earth experiment in Bulgarian and English against the
# GPTQ-W4A16 quantisation of BgGPT-Gemma-3-27B-IT (INSAIT's 27B Bulgarian-
# specialised continued-pretraining of Gemma-3-27B). 4-bit weights fit on
# a single 48 GB GPU, so no tensor parallelism is needed — this avoids the
# silent TP=2 weight-load hang that killed the bf16 attempt (job 224750).

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"
export VLLM_ENGINE_READY_TIMEOUT_S=7200

# BgGPT inherits the Gemma Terms of Use — needs an HF token with the licence accepted.
if [ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ]; then
    export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
fi

source .venv/bin/activate

MODEL="INSAIT-Institute/BgGPT-Gemma-3-27B-IT-GPTQ-W4A16"

echo "Pre-downloading $MODEL weights..."
/ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "[warning] Download failed — model may require HF token with accepted Gemma license"
echo ""

echo "Starting BgGPT-27B (GPTQ W4A16) Bulgarian-prompt experiment..."
echo "Time: $(date)"
echo ""

# Run 1: Bulgarian prompt (the headline run)
echo "=== Run 1/2: Bulgarian prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language bg \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

# Run 2: English baseline on the same model, reusing the loaded server,
# so we can compare BgGPT-27B's English vs Bulgarian maps directly.
echo "=== Run 2/2: English baseline (same model, reused server) ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language en \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --reuse-existing-server

echo ""
echo "BgGPT-27B (GPTQ) Bulgarian experiment completed at $(date)"
