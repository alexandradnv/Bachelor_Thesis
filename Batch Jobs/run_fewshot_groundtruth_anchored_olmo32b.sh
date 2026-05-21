#!/bin/bash
#SBATCH --job-name=blind-earth-gt-anc-olmo32b
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_gt_anc_olmo32b_%j.log
#SBATCH --error=logs/blind_earth_gt_anc_olmo32b_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Few-shot ground-truth + Mannheim anchor, allenai/Olmo-3.1-32B-Instruct (TP=2).

mkdir -p logs
export HF_HOME=/ceph/adinchev/.cache/huggingface
export HF_HUB_CACHE=/ceph/adinchev/.cache/huggingface/hub
export XDG_CACHE_HOME=/ceph/adinchev/.cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_ENGINE_READY_TIMEOUT_S=7200
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
[ -z "$HF_TOKEN" ] && [ -f "/ceph/adinchev/.hf_token" ] && export HF_TOKEN=$(cat /ceph/adinchev/.hf_token)
source .venv/bin/activate

echo "Starting GT+anchor (Olmo-3.1-32B-Instruct, TP=2)  at $(date)"
/ceph/adinchev/experiment/.venv/bin/python fewshot_groundtruth_experiment.py \
  --model "allenai/Olmo-3.1-32B-Instruct" --memory-size 3 --resolution 2 \
  --tensor-parallel-size 2 --anchor mannheim
echo "Done at $(date)"
