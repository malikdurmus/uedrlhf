#!/bin/bash
# slurm
#SBATCH --job-name=finetune
#SBATCH --partition=NvidiaAll
#SBATCH --output=outputs/slurm_logs/%j_out.log # %j will be the Job ID
#SBATCH --error=outputs/slurm_logs/%j_err.log

mkdir -p ../outputs/slurm_logs
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate
#export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
python -u scripts/train/finetune_linear.py 