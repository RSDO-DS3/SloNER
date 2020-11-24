#!/bin/bash
set -euo pipefail

echo "$@"

echo "Starting the BERT process..."
PYTHONPATH=. python src/train/crosloeng.py "$@"
