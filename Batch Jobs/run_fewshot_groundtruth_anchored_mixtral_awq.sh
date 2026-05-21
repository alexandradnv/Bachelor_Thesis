#!/bin/bash
#SBATCH --job-name=blind-earth-gt-anc-mixtral-awq
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_gt_anc_mixtral_awq_%j.log
#SBATCH --error=logs/blind_earth_gt_anc_mixtral_awq_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Few-shot ground-truth + Mannheim anchor, casperhansen/mixtral-instruct-awq (TP=1).

mkdir -p logs
export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
export XDG_CACHE_HOME=/ceph/adinchev/.cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_ENGINE_READY_TIMEOUT_S=7200
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
[ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ] && export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
source .venv/bin/activate

echo "Starting GT+anchor (mixtral-instruct-awq, TP=1)  at $(date)"
/ceph/adinchev/experiment/.venv/bin/python fewshot_groundtruth_experiment.py \
  --model "casperhansen/mixtral-instruct-awq" --memory-size 3 --resolution 2 \
  --tensor-parallel-size 1 --anchor mannheim
echo "Done at $(date)"
