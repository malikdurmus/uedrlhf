#!/bin/bash
# =============================================================
# Step 2: Generate 30k SOLVABLE-ONLY JaxNav environments
# =============================================================
#SBATCH --job-name=gen_solvable
#SBATCH --partition=NvidiaAll
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log

mkdir -p outputs/slurm_logs/v3
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate

# Configure: jaxnav, v3, only_solvable=true
sed -i 's/^environment:.*/environment: jaxnav/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: true/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^llm_provider:.*/llm_provider: gemini/" config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml

echo "=== Config set: jaxnav / v3 / solvable ==="
grep -E '^(environment|data_version|only_solvable|llm_provider|feedback_source)' config.yaml

export XLA_PYTHON_CLIENT_PREALLOCATE=False
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

python -u scripts/generate/generate.py
