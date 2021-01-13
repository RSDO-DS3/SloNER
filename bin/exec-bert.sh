#!/bin/bash
set -euo pipefail

echo "Starting the BERT process..."
PYTHONPATH=. python src/train/crosloeng.py "$@"
# PYTHONPATH=. python src/utils/load_dataset.py "$@"
