#!/bin/bash
# slurm
#SBATCH --job-name=pretrain_array
#SBATCH --partition=NvidiaAll
#SBATCH --array=0-9
#SBATCH --output=outputs/slurm_logs/%A_%a_out.log # %A = array Job ID, %a = task (seed)
#SBATCH --error=outputs/slurm_logs/%A_%a_err.log

mkdir -p ../outputs/slurm_logs
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate
#export XLA_PYTHON_CLIENT_MEM_FRACTION=.90

# One task per seed: seed i of this sweep pairs with seed i of other methods
# for the Wilcoxon signed-rank test.
export SEED=${SLURM_ARRAY_TASK_ID}
export PROJECT_NAME=uedrlhf_train_solvability_pretrain_jaxnav_1m
export RUN_NAME=resnet-pretrain-solvability-seed${SEED}

python -u scripts/train/pretrain_solvability.py
