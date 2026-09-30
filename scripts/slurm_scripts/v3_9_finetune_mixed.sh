#!/bin/bash
# slurm
#SBATCH --job-name=ft_mixed
#SBATCH --partition=NvidiaAll
#SBATCH --array=0-14
#SBATCH --output=outputs/slurm_logs/v3/%A_%a_out.log
#SBATCH --error=outputs/slurm_logs/v3/%A_%a_err.log

# 3 Methods x 5 Runs = 15 Jobs for the Mixed Dataset

mkdir -p outputs/slurm_logs/v3
cd "${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}"
source .venv/bin/activate

export DATASET_PATH="data_v3/jaxnav/difficulty_feedback/compiled_gemini_3_5_flash_len_21990"
DS_TAG="mixed"

# Determine Method (blocks of 5 runs per method)
if [ $SLURM_ARRAY_TASK_ID -lt 5 ]; then
    SCRIPT="scripts/train/finetune_linear.py"
    METHOD_TAG="linear"
elif [ $SLURM_ARRAY_TASK_ID -lt 10 ]; then
    SCRIPT="scripts/train/finetune_diff_lr.py"
    METHOD_TAG="diff_lr"
else
    SCRIPT="scripts/train/finetune_full.py"
    METHOD_TAG="full"
fi

export PRETRAINED_CHECKPOINT_PATH="$PWD/outputs/checkpoints/3zsrto52/epoch_17"
export PROJECT_NAME="gemini35-finetune"
export RUN_NAME="${METHOD_TAG}-${DS_TAG}"

echo "=========================================="
echo "Starting Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset Path: $DATASET_PATH"
echo "Python Script: $SCRIPT"
echo "W&B Project: $PROJECT_NAME"
echo "W&B Run Name: $RUN_NAME"
echo "=========================================="

python -u $SCRIPT
