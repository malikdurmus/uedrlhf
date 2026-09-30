#!/bin/bash
# =============================================================
# Step 8: Compile SOLVABLE dataset into HuggingFace format
# =============================================================
#SBATCH --job-name=comp_solv
#SBATCH --partition=Krater
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/BIG/hf_cache}"
export HF_HOME="${HF_HOME:-$HOME/BIG/hf_cache}"
mkdir -p outputs/slurm_logs/v3
cd "${REPO_ROOT:-$HOME/Desktop/GIT/new/uedrlhf}"
source .venv/bin/activate

# Configure: jaxnav, v3, only_solvable=true (solvable)
sed -i 's/^environment:.*/environment: jaxnav/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: true/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml

echo "=== Config set: jaxnav / v3 / solvable — compile ==="

python -u scripts/dataset/compile.py
