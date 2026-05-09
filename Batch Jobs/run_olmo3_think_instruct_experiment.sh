#!/bin/bash

#SBATCH --job-name=blind-earth-olmo3-7b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gpus-per-socket=2
#SBATCH --sockets-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_olmo3_7b_%j.log
#SBATCH --error=logs/blind_earth_olmo3_7b_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"

# 7B models load much faster than 14B — use a 1-hour engine timeout
# (the 7200s default caused 221367 to hang for 2h per model on failed starts)
export VLLM_ENGINE_READY_TIMEOUT_S=3600

export HF_HUB_ENABLE_HF_TRANSFER=0

source .venv/bin/activate

MODELS=(
    "allenai/Olmo-3-7B-Think"
    "allenai/Olmo-3-7B-Instruct"
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

echo "Starting blind earth experiment (OLMo-3 7B: Think + Instruct)..."
echo "Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python blind_model_experiment.py \
  --models "${MODELS[@]}" \
  --resolution 2 \
  --workers 32 \
  --tensor-parallel-size 1

echo ""
echo "Experiment completed at $(date)"
