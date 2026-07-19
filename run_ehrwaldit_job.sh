#!/bin/bash

# Exit immediately if any step fails
set -e

echo "=== Starting Environment Setup ==="
source .venv/bin/activate

# Configure: jaxnav, v3, only_solvable=false
sed -i 's/^environment:.*/environment: jaxnav/' config.yaml
sed -i 's/^  only_solvable:.*/  only_solvable: false/' config.yaml
sed -i "s/^data_version:.*/data_version: \"v3\"/" config.yaml
sed -i "s/^llm_provider:.*/llm_provider: gemini/" config.yaml
sed -i "s/^feedback_source:.*/feedback_source: \"gemini-3.5-flash\"/" config.yaml

echo "=== Config set: jaxnav / v3 / mixed ==="
grep -E '^(environment|data_version|only_solvable|llm_provider|feedback_source)' config.yaml

# JAX Memory Configuration
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# Generate a clean timestamp for the log file name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="run_output_${TIMESTAMP}.log"

echo "=== Launching Script (Logging to ${LOG_FILE}) ==="

# Execute Python, combining stdout and stderr into the timestamped log file
python -u scripts/generate/generate.py 2>&1 | tee "$LOG_FILE"
