#!/bin/bash

set -e
DEST=$1
PORT=${2:-22}
if [ -z "$DEST" ]; then echo "Usage: bash cloud/push_to_vm.sh user@ip [port]"; exit 1; fi
RSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

cd "$(dirname "$0")/.."

echo "=== Syncing code (excluding data, logs, creds) ==="
rsync -avz -e "$RSH" \
  --exclude '.venv' --exclude 'wandb' --exclude 'outputs' --exclude 'slurm_logs' \
  --exclude 'data' --exclude 'data_v2' --exclude 'data_v3' --exclude 'datav22' \
  --exclude 'dataset_1M' --exclude 'dataset_balances' --exclude 'env_baselines' \
  --exclude 'rendered_levels' --exclude 'orbaxexport' --exclude 'testing_renderer' \
  --exclude '*.jsonl' --exclude '*.log' \
  --exclude 'gen-lang-client-*.json' --exclude 'keremproject-*.json' --exclude 'migmg_acc_project-*.json' \
  ./ "$DEST":uedrlhf/

echo "=== Syncing datasets + pretrained checkpoint ==="
rsync -avzR --progress -e "$RSH" \
  data_v3/xland_minigrid/ruleset_medium/difficulty_feedback/compiled_gemini_3_5_flash_len_7048 \
  data_v3/jaxnav/difficulty_feedback_solvable/compiled_gemini_3_5_flash_len_21989 \
  outputs/checkpoints/3zsrto52 \
  "$DEST":uedrlhf/

echo "=== Copying wandb API key (if logged in here) ==="
KEY=$(awk '/api\.wandb\.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc 2>/dev/null)
if [ -n "$KEY" ]; then
  $RSH "$DEST" "grep -q WANDB_API_KEY ~/.bashrc 2>/dev/null || echo 'export WANDB_API_KEY=$KEY' >> ~/.bashrc"
  echo "wandb key installed on VM."
else
  echo "No wandb key found in ~/.netrc — run 'wandb login' on the VM before training."
fi

echo ""
echo "Done. Now on the VM run:"
echo "  bash uedrlhf/cloud/setup_vm.sh"
