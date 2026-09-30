#!/bin/bash
# slurm
#SBATCH --job-name=env_gen
#SBATCH --partition=NvidiaAll
#SBATCH --output=outputs/slurm_logs/%j_out.log # %j will be the Job ID
#SBATCH --error=outputs/slurm_logs/%j_err.log

mkdir -p outputs/slurm_logs
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate
#export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
python -u scripts/generate/gen_validity_env.py \
    --num_envs 1000000 \
    --batch_size 5000 \
    --shard_size 50000 \
    --out_dir /home/d/durmusy/Desktop/GIT/new/uedrlhf/dataset_1M

echo "Job Finished!"