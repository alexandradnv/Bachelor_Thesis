#!/bin/bash

#SBATCH --job-name=blind-earth-fewshot-gt-qwen32b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_fewshot_gt_qwen32b_%j.log
#SBATCH --error=logs/blind_earth_fewshot_gt_qwen32b_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Few-shot ground-truth, Qwen/Qwen2.5-32B-Instruct (full precision).
# Needs TP=2 across two 48 GB GPUs.

mkdir -p logs
export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
export XDG_CACHE_HOME=/ceph/adinchev/.cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_ENGINE_READY_TIMEOUT_S=7200
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
[ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ] && export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
source .venv/bin/activate

MODEL="Qwen/Qwen2.5-32B-Instruct"

echo "Starting few-shot ground-truth ($MODEL, TP=2)  at $(date)"
/ceph/adinchev/experiment/.venv/bin/python fewshot_groundtruth_experiment.py \
  --model "$MODEL" --memory-size 3 --resolution 2 --tensor-parallel-size 2
echo "Done at $(date)"
