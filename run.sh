#!/bin/bash
# slurm
#SBATCH --job-name=cnntrain
#SBATCH --partition=NvidiaAll
#SBATCH --output=slurm_logs/%j_out.log # %j will be the Job ID
#SBATCH --error=slurm_logs/%j_err.log

mkdir -p slurm_logs
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate
#export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
python -u train/train_pixels.py