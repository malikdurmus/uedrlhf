#!/bin/bash
# Local replacement for `sbatch` on a cloud VM: runs an existing SLURM array
# script UNMODIFIED, emulating SLURM_ARRAY_TASK_ID.
#
# Usage: bash cloud/sbatch_local.sh <slurm_script> [max_parallel_tasks]
#   e.g. bash cloud/sbatch_local.sh scripts/slurm_scripts/train_cnn_medium_normal.sh 2
#
# max_parallel_tasks defaults to 1 (sequential). With
# XLA_PYTHON_CLIENT_PREALLOCATE=false several tasks can share one GPU;
# on a multi-GPU box tasks are spread round-robin via CUDA_VISIBLE_DEVICES.
set -e
SCRIPT=$1
PAR=${2:-1}
if [ -z "$SCRIPT" ]; then echo "Usage: bash cloud/sbatch_local.sh <slurm_script> [max_parallel]"; exit 1; fi
cd "$(dirname "$0")/.."

SPEC=$(grep -oE -- '--array=[0-9]+(-[0-9]+)?' "$SCRIPT" | cut -d= -f2)
SPEC=${SPEC:-0}
LO=${SPEC%-*}; HI=${SPEC#*-}
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}; [ "$NGPU" -eq 0 ] && NGPU=1
export SLURM_ARRAY_JOB_ID=$$
mkdir -p outputs/slurm_logs/v3
NAME=$(basename "$SCRIPT" .sh)

echo "Running $SCRIPT: array tasks $LO..$HI, $PAR in parallel, $NGPU GPU(s)"
i=0
for T in $(seq "$LO" "$HI"); do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  GPU=$(( i % NGPU ))
  LOGB="outputs/slurm_logs/v3/${NAME}_local_${T}"
  echo ">>> task $T on GPU $GPU  (logs: ${LOGB}_out.log)"
  SLURM_ARRAY_TASK_ID=$T CUDA_VISIBLE_DEVICES=$GPU \
    bash "$SCRIPT" > "${LOGB}_out.log" 2> "${LOGB}_err.log" &
  i=$((i+1))
done
wait
echo "All $((HI - LO + 1)) tasks finished. Logs in outputs/slurm_logs/v3/"
