#!/bin/bash
# =============================================================
# Step 2: Submit MIXED envs to Gemini 3.5 Flash batch API (difficult - low thinking)
# =============================================================
#SBATCH --job-name=bat_x_difficult_low
#SBATCH --partition=NvidiaAll
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log
mkdir -p outputs/slurm_logs/v3
cd "${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}"
source .venv/bin/activate

sed -i 's/^environment:.*/environment: xland_minigrid/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: false/' config.yaml
sed -i 's/^  ruleset:.*/  ruleset: difficult/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^llm_provider:.*/llm_provider: gemini/" config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml
sed -i '/^  thinking_level:.*/d' config.yaml
sed -i '/^gemini:/a \ \ thinking_level: "low"' config.yaml

echo "=== Config set: xland_minigrid / v3 / mixed / difficult / low — batch submit ==="

# Load environment variables from .env file
set -a; source .env; set +a

python -u scripts/feedback/llm_batch_submit.py --model gemini-3.5-flash
