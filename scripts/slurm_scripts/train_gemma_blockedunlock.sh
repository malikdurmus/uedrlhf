#!/bin/bash
# Gemma-labeled BlockedUnlockPickUp CNN runs (label-quality evidence for the
# Gemini-vs-Gemma comparison; pre-registered THESIS_NOTES 2026-09-02).
# Old-format dataset served via the verified legacy-float loader branch.
#SBATCH --job-name=gemma_bup
#SBATCH --partition=NvidiaAll
#SBATCH --array=0-4
#SBATCH --output=outputs/slurm_logs/v3/gemma_bup_%a_%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/gemma_bup_%a_%j_err.log
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate
export DATASET_PATH=data_v2/xland_minigrid/MiniGrid-BlockedUnlockPickUp/difficulty_feedback/compiled_gemma4_e2b_len_21992
export PROJECT_NAME=tvt2-n-gemma4-direct-blockedunlock
export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME=cnn-gemma-bup-seed$SEED
export AUGMENT_SYMMETRY=0
python -u scripts/train/train_pixels_fast.py
