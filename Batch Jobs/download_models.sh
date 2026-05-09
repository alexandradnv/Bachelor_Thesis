#!/bin/bash

#SBATCH --job-name=download-models
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/download_models_%j.log
#SBATCH --error=logs/download_models_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
mkdir -p "$HF_HOME"

source .venv/bin/activate

echo "Starting model downloads at $(date)"
echo "HF_HOME=$HF_HOME"
echo ""

MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "allenai/Olmo-3-7B-Think"
)

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Downloading: $MODEL"
    echo "Started: $(date)"
    /ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
"
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "Done: $MODEL at $(date)"
    else
        echo "FAILED: $MODEL (exit code $STATUS)"
    fi
    echo ""
done

echo "All downloads finished at $(date)"
