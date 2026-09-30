#!/bin/bash
# =============================================================
# Step 4: Compile MIXED dataset into HuggingFace format (medium)
# =============================================================
#SBATCH --job-name=com_x_medium
#SBATCH --partition=Krater
#SBATCH --output=outputs/slurm_logs/v3/%j_out.log
#SBATCH --error=outputs/slurm_logs/v3/%j_err.log

export HF_DATASETS_CACHE="/home/d/durmusy/BIG/hf_cache"
export HF_HOME="/home/d/durmusy/BIG/hf_cache"
mkdir -p outputs/slurm_logs/v3
cd ~/Desktop/GIT/new/uedrlhf
source .venv/bin/activate

sed -i 's/^environment:.*/environment: xland_minigrid/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: false/' config.yaml
sed -i 's/^  ruleset:.*/  ruleset: medium/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml
sed -i '/^  thinking_level:.*/d' config.yaml

echo "=== Config set: xland_minigrid / v3 / mixed / medium — compile ==="

python -u scripts/dataset/compile.py
