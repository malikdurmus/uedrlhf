#!/bin/bash
# =============================================================
# Step 1: Generate MIXED XLand-MiniGrid-R4-13x13 environments (easy)
# =============================================================
#SBATCH --job-name=gen_x_easy
#SBATCH --partition=NvidiaAll
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log

mkdir -p outputs/slurm_logs/v3
cd "${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}"
source .venv/bin/activate

# Configure: xland_minigrid, v3, only_solvable=false, ruleset=easy
sed -i 's/^environment:.*/environment: xland_minigrid/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: false/' config.yaml
sed -i 's/^  ruleset:.*/  ruleset: easy/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^llm_provider:.*/llm_provider: gemini/" config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml

echo "=== Config set: xland_minigrid / v3 / mixed / easy ==="
grep -E '^(environment|data_version|only_solvable|llm_provider|feedback_source|  ruleset)' config.yaml

export XLA_PYTHON_CLIENT_PREALLOCATE=False
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

python -u scripts/generate/generate.py
