#!/bin/bash

#SBATCH --job-name=download-qwen32b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/download_qwen32b_%j.log
#SBATCH --error=logs/download_qwen32b_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

mkdir -p logs

source .venv/bin/activate

export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

echo "Downloading Qwen2.5-32B-Instruct weights..."
echo "Time: $(date)"
echo "Cache dir: $HF_HUB_CACHE"
echo ""

/ceph/adinchev/experiment/.venv/bin/huggingface-cli download Qwen/Qwen2.5-32B-Instruct

echo ""
echo "Download completed at $(date)"
