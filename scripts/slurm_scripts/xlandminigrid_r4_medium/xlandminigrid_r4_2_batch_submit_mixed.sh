#!/bin/bash
# =============================================================
# Step 2: Submit MIXED envs to Gemini 2.5 Flash batch API (medium)
# =============================================================
#SBATCH --job-name=bat_x_medium
#SBATCH --partition=NvidiaAll
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log
mkdir -p outputs/slurm_logs/v3
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate

sed -i 's/^environment:.*/environment: xland_minigrid/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: false/' config.yaml
sed -i 's/^  ruleset:.*/  ruleset: medium/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^llm_provider:.*/llm_provider: gemini/" config.yaml
sed -i '/^  thinking_level:.*/d' config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml

echo "=== Config set: xland_minigrid / v3 / mixed / medium — batch submit ==="

# Load environment variables from .env file
set -a; source .env; set +a

python -u scripts/feedback/llm_batch_submit.py --model gemini-3.5-flash
