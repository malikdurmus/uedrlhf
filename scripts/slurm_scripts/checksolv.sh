#!/bin/bash
# =============================================================
# Step 7: Compile MIXED dataset into HuggingFace format
# =============================================================
#SBATCH --job-name=shcsolv
#SBATCH --partition=NvidiaAll
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log
mkdir -p outputs/slurm_logs/v3
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate


python -u scripts/analysis/check_solvability_v3.py