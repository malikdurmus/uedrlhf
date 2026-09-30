#!/bin/bash
#SBATCH --job-name=cnn_low_aug
#SBATCH --partition=NvidiaAll
#SBATCH --array=0-4
#SBATCH --output=outputs/slurm_logs/v3/%x_%A_%a_out.log
#SBATCH --error=outputs/slurm_logs/v3/%x_%A_%a_err.log

mkdir -p outputs/slurm_logs/v3
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async

# One task per seed; identical recipe to the unaugmented difficult-low runs,
# except AUGMENT_SYMMETRY=1 (random rot90/flip on train batches only).
export SEED=${SLURM_ARRAY_TASK_ID}
export AUGMENT_SYMMETRY=1
export DATASET_PATH="data_v3/xland_minigrid/ruleset_difficult/difficulty_feedback_low/compiled_gemini_3_5_flash_len_21973"
export PROJECT_NAME="gemini35-direct-xminigrid-r41313-difficult-low-augmented"
export RUN_NAME="cnn_difficult_low_augmented_seed${SEED}"

echo "=== Starting Training (difficult ruleset, low thinking, symmetry augmentation) - Seed ${SEED} ==="
echo "Dataset: ${DATASET_PATH}"
echo "Project: ${PROJECT_NAME}"
echo "Run Name: ${RUN_NAME}"

python -u scripts/train/train_pixels.py
