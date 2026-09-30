#!/bin/bash
# slurm
#SBATCH --job-name=finetune_array
#SBATCH --partition=NvidiaAll
#SBATCH --array=0-29
#SBATCH --output=outputs/slurm_logs/v3/%A_%a_out.log
#SBATCH --error=outputs/slurm_logs/v3/%A_%a_err.log

# Creates 30 parallel jobs (0 to 29) to cover:
# 2 Datasets x 3 Methods x 5 Runs = 30 Jobs

mkdir -p outputs/slurm_logs/v3
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate

# 1. Determine Dataset (first 15 jobs are mixed, last 15 are solvable)
if [ $SLURM_ARRAY_TASK_ID -lt 15 ]; then
    export DATASET_PATH="data_v3/jaxnav/difficulty_feedback/compiled_gemini_3_5_flash_len_21990"
    DS_TAG="mixed"
else
    export DATASET_PATH="data_v3/jaxnav/difficulty_feedback_solvable/compiled_gemini_3_5_flash_len_21989"
    DS_TAG="solvable"
fi

# 2. Determine Method (blocks of 5 runs per method)
MOD_VAL=$((SLURM_ARRAY_TASK_ID % 15))

if [ $MOD_VAL -lt 5 ]; then
    SCRIPT="scripts/train/finetune_linear.py"
    METHOD_TAG="linear"
elif [ $MOD_VAL -lt 10 ]; then
    SCRIPT="scripts/train/finetune_diff_lr.py"
    METHOD_TAG="diff_lr"
else
    SCRIPT="scripts/train/finetune_full.py"
    METHOD_TAG="full"
fi

# 3. Export global variables for the python scripts to read
export PRETRAINED_CHECKPOINT_PATH="/home/d/durmusy/Desktop/GIT/new/uedrlhf/outputs/checkpoints/3zsrto52/epoch_17"
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
