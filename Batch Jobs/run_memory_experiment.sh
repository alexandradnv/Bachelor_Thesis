#!/bin/bash

#SBATCH --job-name=blind-earth-memory
#SBATCH --partition=gpu-vram-48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/blind_earth_memory_%j.log
#SBATCH --error=logs/blind_earth_memory_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Memory-conditioned blind-earth experiment.
#
# For each lat/lon (visited in scan order) the model is shown the last 3
# (coord, verdict, P(Land)) tuples it produced.  The chain is fully
# sequential — one query at a time — so the model genuinely sees its own
# most-recent answers.
#
# We start with the 3B Qwen for a sanity-check run.  Generates:
#   Generated models/Qwen_Qwen2.5-3B-Instruct_memory3.png
#   Generated models/Qwen_Qwen2.5-3B-Instruct_memory3_data.json
#   MSEs/memory3/mse_summary.{json,csv}
#
# Wall time estimate: ~16,200 queries × ~80–150 ms each on a single 48 GB
# GPU ≈ 25–45 min once vLLM is warm.  12 h SLURM limit leaves plenty of
# headroom for slow ceph weight loading.

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

echo "Starting memory-conditioned blind-earth experiment"
echo "  Model:        $MODEL"
echo "  Memory size:  $MEMORY_SIZE previous (coord, P(Land)) tuples"
echo "  Resolution:   ${RESOLUTION}° (fully sequential single chain)"
echo "  Time:         $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python memory_experiment.py \
  --model "$MODEL" \
  --memory-size "$MEMORY_SIZE" \
  --resolution "$RESOLUTION" \
  --tensor-parallel-size 1

echo ""
echo "Memory experiment completed at $(date)"
