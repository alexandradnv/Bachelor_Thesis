#!/bin/bash

#SBATCH --job-name=blind-earth-diverse-v4
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=1
#SBATCH --sockets-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_diverse_v4_%j.log
#SBATCH --error=logs/blind_earth_diverse_v4_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Fourth diverse-models batch. All four fit on a single 48 GB GPU, so TP=1
# (more reliable than TP=2, which caused the engine-core timeout in v3).
#
#   01-ai/Yi-1.5-9B-Chat                                 (9B  - Yi 1.5, upgrade from Yi-6B)
#   INSAIT-Institute/BgGPT-Gemma-3-27B-IT-GPTQ-W4A16    (27B - Gemma-3 BG fine-tune, 4-bit)
#   internlm/internlm2_5-7b-chat                         (7B  - InternLM, Shanghai AI Lab)
#   HuggingFaceH4/zephyr-7b-beta                         (7B  - classic Mistral-7B fine-tune)

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"
export VLLM_ENGINE_READY_TIMEOUT_S=7200

# BgGPT inherits the Gemma Terms of Use — needs an HF token with the licence accepted.
if [ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ]; then
    export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
fi

source .venv/bin/activate

MODELS=(
    "01-ai/Yi-1.5-9B-Chat"
    "INSAIT-Institute/BgGPT-Gemma-3-27B-IT-GPTQ-W4A16"
    "internlm/internlm2_5-7b-chat"
    "HuggingFaceH4/zephyr-7b-beta"
)

echo "Pre-downloading model weights to cache..."
for MODEL in "${MODELS[@]}"; do
    echo "  -> $MODEL"
    /ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "     [warning] Download failed for $MODEL"
done
echo "Pre-download complete."
echo ""

echo "Starting blind earth experiment (diverse models v4)..."
echo "Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "${MODELS[@]}" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

echo ""
echo "Experiment completed at $(date)"
