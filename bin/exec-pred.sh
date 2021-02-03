#!/bin/bash
set -euo pipefail

echo "Starting the BERT prediction process..."
PYTHONPATH=. python src/eval/predict.py "$@"
