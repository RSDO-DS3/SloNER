#!/bin/bash
set -euo pipefail

echo "Starting the training process..."
PYTHONPATH=. python src/train/crosloeng.py
