#!/bin/bash
# =============================================================
# Step 3: Collect MIXED batch results from Gemini (medium)
# =============================================================
# USAGE: sbatch xlandminigrid_r4_3_batch_collect_mixed.sh --job batches/YOUR_JOB_ID
#
#SBATCH --job-name=col_x_medium
#SBATCH --partition=Krater
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log
mkdir -p outputs/slurm_logs/v3
cd "${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}"
source .venv/bin/activate

sed -i 's/^environment:.*/environment: xland_minigrid/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: false/' config.yaml
sed -i 's/^  ruleset:.*/  ruleset: medium/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^llm_provider:.*/llm_provider: gemini/" config.yaml
sed -i '/^  thinking_level:.*/d' config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml

echo "=== Config set: xland_minigrid / v3 / mixed / medium — batch collect ==="

# Load environment variables from .env file
set -a; source .env; set +a

# Pass through the --job argument from sbatch command line
uv run scripts/feedback/llm_batch_collect.py --job projects/2329960787/locations/global/batchPredictionJobs/8267730131183730688 --model gemini-3.5-flash
