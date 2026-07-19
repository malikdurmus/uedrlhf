#!/bin/bash
# =============================================================
# Step 4: Submit SOLVABLE envs to Gemini 2.5 Flash batch API
# =============================================================
#SBATCH --job-name=batch_solv
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

echo "=== Config set: jaxnav / v3 / solvable — batch submit ==="

# Load environment variables from .env file
set -a; source .env; set +a

python -u scripts/feedback/llm_batch_submit.py --model gemini-3.5-flash
