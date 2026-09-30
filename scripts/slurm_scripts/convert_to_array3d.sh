#!/bin/bash
# =============================================================
# Re-encode a compiled dataset with Array3D(uint8) schema
# Usage: sbatch scripts/slurm_scripts/convert_to_array3d.sh <src_dataset_dir>
# =============================================================
#SBATCH --job-name=conv_arr
#SBATCH --partition=Krater
#SBATCH --output=outputs/slurm_logs/v3/%x_%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%x_%j_err.log

export HF_DATASETS_CACHE="/home/d/durmusy/BIG/hf_cache"
export HF_HOME="/home/d/durmusy/BIG/hf_cache"
mkdir -p outputs/slurm_logs/v3
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate

SRC=$1
echo "=== Converting: $SRC ==="
python -u scripts/dataset/convert_to_array3d.py "$SRC"
