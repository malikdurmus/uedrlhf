#!/bin/bash
# =============================================================
# Step 5: Collect MIXED batch results from Gemini
# =============================================================
# USAGE: sbatch v3_5_batch_collect_mixed.sh --job batches/YOUR_JOB_ID
#
#SBATCH --job-name=coll_mixed
#SBATCH --partition=Krater
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log
mkdir -p outputs/slurm_logs/v3
cd "${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}"
source .venv/bin/activate

# Configure: jaxnav, v3, only_solvable=false (mixed)
sed -i 's/^environment:.*/environment: jaxnav/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: false/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^llm_provider:.*/llm_provider: gemini/" config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml

echo "=== Config set: jaxnav / v3 / mixed — batch collect ==="

# Load environment variables from .env file
set -a; source .env; set +a

# Pass through the --job argument from sbatch command line
uv run scripts/feedback/llm_batch_collect.py --job projects/962036078429/locations/global/batchPredictionJobs/8186338233181798400 --model gemini-3.5-flash