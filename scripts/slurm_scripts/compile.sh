#!/bin/bash
# slurm
#SBATCH --job-name=compileds
#SBATCH --partition=Krater
#SBATCH --output=outputs/slurm_logs/%j_out.log # %j will be the Job ID
#SBATCH --error=outputs/slurm_logs/%j_err.log

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/BIG/hf_cache}"
export HF_HOME="${HF_HOME:-$HOME/BIG/hf_cache}"
mkdir -p ../outputs/slurm_logs
cd "${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}"
source .venv/bin/activate
#export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
python -u scripts/dataset/compile.py