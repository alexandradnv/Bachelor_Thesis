#!/bin/bash

#SBATCH --job-name=blind-earth-mem-modes-7b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_mem_modes_7b_%j.log
#SBATCH --error=logs/blind_earth_mem_modes_7b_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Memory-experiment variants on Qwen2.5-7B-Instruct.  See run_memory_modes_3b.sh
# for the protocol description — this is the identical job, 7B instead of 3B.
# 7B is the model that fully collapsed to "all-water" under scan-order memory,
# so this is the most interesting size to re-run with the two fixes.

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

echo "Starting memory-modes experiment (7B): shuffle + perrowseed"
echo "  Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python memory_experiment.py \
  --model "$MODEL" \
  --modes shuffle perrowseed \
  --memory-size 3 \
  --resolution 2 \
  --tensor-parallel-size 1

echo ""
echo "Memory-modes (7B) completed at $(date)"
