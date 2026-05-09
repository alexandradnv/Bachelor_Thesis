#!/bin/bash

#SBATCH --job-name=blind-earth-gemma3-27b-bggpt-languages
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_gemma3_27b_bggpt_languages_%j.log
#SBATCH --error=logs/blind_earth_gemma3_27b_bggpt_languages_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Gemma3-27B language-bias experiment via the BgGPT GPTQ W4A16 quant.
# This is the only Gemma3-27B variant accessible without further HF license
# acceptance — INSAIT's continued-pretraining of google/gemma-3-27b-it,
# already cached on disk and proven to run. 4-bit weights fit on a single
# 48GB GPU, so no TP=2 (which has been unstable on this cluster).
# Note: the BG fine-tuning will boost the Bulgarian map relative to vanilla
# Gemma3-27B; treat results in that language with caution.

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

echo "Starting Gemma3-27B (BgGPT GPTQ W4A16) language-bias experiment..."
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
  --tensor-parallel-size 1

# Runs 2-5 reuse the already-loaded server so the model only loads once.
echo "=== Run 2/5: Bulgarian prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language bg \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --reuse-existing-server

echo "=== Run 3/5: English prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language en \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --reuse-existing-server

echo "=== Run 4/5: German prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language de \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --reuse-existing-server

echo "=== Run 5/5: Thai prompt ==="
/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "$MODEL" \
  --language th \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1 \
  --reuse-existing-server

echo ""
echo "All Gemma3-27B (BgGPT) language runs completed at $(date)"
