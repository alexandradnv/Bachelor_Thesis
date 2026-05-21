#!/bin/bash

#SBATCH --job-name=blind-earth-fewshot-gt-7b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_fewshot_gt_7b_%j.log
#SBATCH --error=logs/blind_earth_fewshot_gt_7b_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Few-shot ground-truth experiment, Qwen2.5-7B-Instruct.  Same protocol as
# the 3B variant: scan-order, sequential single chain, 3 previous coords
# with GROUND-TRUTH labels in every prompt.

mkdir -p logs

export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
export XDG_CACHE_HOME=/ceph/adinchev/.cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_ENGINE_READY_TIMEOUT_S=7200
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

if [ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ]; then
    export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
fi

source .venv/bin/activate

MODEL="Qwen/Qwen2.5-7B-Instruct"
MEMORY_SIZE=3
RESOLUTION=2

echo "Starting few-shot ground-truth experiment (7B)"
echo "  Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python fewshot_groundtruth_experiment.py \
  --model "$MODEL" \
  --memory-size "$MEMORY_SIZE" \
  --resolution "$RESOLUTION" \
  --tensor-parallel-size 1

echo ""
echo "Few-shot ground-truth (7B) completed at $(date)"
