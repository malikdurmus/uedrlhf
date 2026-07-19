#!/bin/bash
# =============================================================
# Step 6: Collect SOLVABLE batch results from Gemini
# =============================================================
# USAGE: sbatch v3_6_batch_collect_solvable.sh --job batches/YOUR_JOB_ID
#
#SBATCH --job-name=coll_solv
#SBATCH --partition=Krater
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log
mkdir -p outputs/slurm_logs/v3
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate

# Configure: jaxnav, v3, only_solvable=true (solvable)
sed -i 's/^environment:.*/environment: jaxnav/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: true/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^llm_provider:.*/llm_provider: gemini/" config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml

echo "=== Config set: jaxnav / v3 / solvable — batch collect ==="

# Load environment variables from .env file
set -a; source .env; set +a

# Pass through the --job argument from sbatch command line
python -u scripts/feedback/llm_batch_collect.py --model gemini-3.5-flash "$@"
