#!/bin/bash
#SBATCH --job-name=cnn_jaxnav_mixed
#SBATCH --partition=NvidiaAll
#SBATCH --array=0-9
#SBATCH --output=outputs/slurm_logs/v3/%x_%A_%a_out.log
#SBATCH --error=outputs/slurm_logs/v3/%x_%A_%a_err.log

mkdir -p outputs/slurm_logs/v3
cd "${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}"
source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async

# One task per seed: seed i pairs with seed i of other methods for the Wilcoxon test
export SEED=${SLURM_ARRAY_TASK_ID}
export DATASET_PATH="data_v3/jaxnav/difficulty_feedback/compiled_gemini_3_5_flash_len_21990"
export PROJECT_NAME="gemini35-direct-jaxnav-solvable-unsolvable-mixed"
export RUN_NAME="cnn_compiled_gemini_3_5_flash_len_21990_seed${SEED}"

echo "=== Starting Training (JaxNav mixed solvable/unsolvable) - Seed ${SEED} ==="
echo "Dataset: ${DATASET_PATH}"
echo "Project: ${PROJECT_NAME}"
echo "Run Name: ${RUN_NAME}"

python -u scripts/train/train_pixels.py
