#!/bin/bash

#SBATCH --job-name=blind-earth-compute-mses
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adinchev@informatik.uni-mannheim.de
#SBATCH --output=logs/compute_all_mses_%j.log
#SBATCH --error=logs/compute_all_mses_%j.err
#SBATCH --chdir=/ceph/adinchev/experiment

# Recompute MSE vs ground truth for every saved blind-earth map
# (Generated models/*_data.json) and write a consolidated ranking to
# MSEs/all_mses.{csv,json}. CPU-only — uses global_land_mask, no model loading.

mkdir -p logs

source .venv/bin/activate

echo "Computing MSEs for all generated maps..."
echo "Time: $(date)"
echo ""

/ceph/adinchev/experiment/.venv/bin/python compute_all_mses.py

echo ""
echo "Done at $(date)"
