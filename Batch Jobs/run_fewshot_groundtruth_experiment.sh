#!/bin/bash

#SBATCH --job-name=blind-earth-fewshot-gt
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_fewshot_gt_%j.log
#SBATCH --error=logs/blind_earth_fewshot_gt_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Controlled counterpart to run_memory_experiment.sh.
#
# Same model, same memory-size, same scan order, same prompt structure —
# only the in-context labels change from "the model's own past P(Land)"
# to "ground-truth verdicts" looked up via global_land_mask.
#
# Expected runtime: ~6 minutes on a single 48 GB GPU once vLLM is warm
# (identical to the 3B memory run since it's the same workload).

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
MEMORY_SIZE=3
RESOLUTION=2

echo "Pre-downloading $MODEL weights..."
/ceph/adinchev/experiment/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='$MODEL', cache_dir=os.environ['HF_HOME'] + '/hub')
" || echo "[warning] Download failed (already cached?)"
echo ""

echo "Starting few-shot ground-truth experiment"
echo "  Model:        $MODEL"
echo "  Context:      $MEMORY_SIZE previous coords with GROUND-TRUTH labels"
echo "  Resolution:   ${RESOLUTION}° (fully sequential single chain)"
echo "  Time:         $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python fewshot_groundtruth_experiment.py \
  --model "$MODEL" \
  --memory-size "$MEMORY_SIZE" \
  --resolution "$RESOLUTION" \
  --tensor-parallel-size 1

echo ""
echo "Few-shot ground-truth experiment completed at $(date)"
