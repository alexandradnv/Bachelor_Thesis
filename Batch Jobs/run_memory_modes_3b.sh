#!/bin/bash

#SBATCH --job-name=blind-earth-mem-modes-3b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_mem_modes_3b_%j.log
#SBATCH --error=logs/blind_earth_mem_modes_3b_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Memory-experiment variants on Qwen2.5-3B-Instruct.
#
# Runs back-to-back inside one Python process so the vLLM server only starts
# once per job:
#   1. --mode shuffle      —  random global visit order; tests whether the
#                             scan-order snowball is what locked the model in.
#   2. --mode perrowseed   —  scan order but memory is reset to 3 fixed
#                             ground-truth anchor coords (Mannheim=Land,
#                             Pacific=Water, Tokyo=Land) at the start of every
#                             row; tests whether periodic re-anchoring breaks
#                             the lock-in.
#
# Outputs:
#   Generated models/Qwen_Qwen2.5-3B-Instruct_memory3_shuffled.{png,json}
#   Generated models/Qwen_Qwen2.5-3B-Instruct_memory3_perrowseed.{png,json}
#   MSEs/memory3_shuffle/mse_summary.{json,csv}
#   MSEs/memory3_perrowseed/mse_summary.{json,csv}

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

MODEL="Qwen/Qwen2.5-3B-Instruct"

echo "Starting memory-modes experiment (3B): shuffle + perrowseed"
echo "  Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python memory_experiment.py \
  --model "$MODEL" \
  --modes shuffle perrowseed \
  --memory-size 3 \
  --resolution 2 \
  --tensor-parallel-size 1

echo ""
echo "Memory-modes (3B) completed at $(date)"
