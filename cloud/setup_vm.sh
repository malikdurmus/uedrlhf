#!/bin/bash
# Run this ON THE CLOUD VM, once. Needs no configuration.
# Recreates the exact environment: python env via uv.lock, and the absolute
# path your slurm scripts cd into (~/Desktop/GIT/new/uedrlhf) as a symlink.
set -e
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD

# your slurm scripts do `cd ~/Desktop/GIT/new/uedrlhf` — make that path exist here
mkdir -p ~/Desktop/GIT/new
[ -e ~/Desktop/GIT/new/uedrlhf ] || ln -s "$REPO_ROOT" ~/Desktop/GIT/new/uedrlhf

# uv installs into ~/.local/bin and manages its own python 3.12 — no sudo needed
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "=== Installing environment from uv.lock (takes a few minutes) ==="
uv sync --frozen

mkdir -p outputs/slurm_logs/v3 outputs/checkpoints

echo "=== Checking JAX sees the GPU ==="
.venv/bin/python -c "import jax; print('JAX devices:', jax.devices())"

echo ""
echo "Setup complete. Run experiments with, e.g.:"
echo "  cd $REPO_ROOT"
echo "  bash cloud/sbatch_local.sh scripts/slurm_scripts/v3_9_finetune_solvable.sh 3"
