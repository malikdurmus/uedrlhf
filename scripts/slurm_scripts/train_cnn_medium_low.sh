#!/bin/bash
#SBATCH --job-name=cnn_low
#SBATCH --partition=NvidiaAll
#SBATCH --array=1-4
#SBATCH --output=outputs/slurm_logs/v3/%x_%A_%a_out.log
#SBATCH --error=outputs/slurm_logs/v3/%x_%A_%a_err.log

mkdir -p outputs/slurm_logs/v3
cd "${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}"
source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async

export DATASET_PATH="data_v3/xland_minigrid/ruleset_medium/difficulty_feedback_low/compiled_gemini_3_5_flash_len_21980"
export PROJECT_NAME="gemini35-direct-xminigrid-r41313-low"
export RUN_NAME="compiled_gemini_3_5_flash_len_21980"

echo "=== Starting Training (Low Thinking Dataset) - Run ${SLURM_ARRAY_TASK_ID} ==="
echo "Dataset: ${DATASET_PATH}"
echo "Project: ${PROJECT_NAME}"
echo "Run Name: ${RUN_NAME}"

python -u scripts/train/train_pixels.py
