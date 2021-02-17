#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/NER-BERT-pred-%J.out
#SBATCH --error=logs/NER-BERT-pred-%J.err
#SBATCH --job-name="NER-BERT-pred"

set -euo pipefail

CONTAINER_IMAGE_PATH="$PWD/containers/pytorch-image.sqfs"

echo "$SLURM_JOB_ID -> Predicting from the model..."

# the following command opens a bash terminal of an already existing container
# with the current directory (.) mounted
srun \
    --container-image "$CONTAINER_IMAGE_PATH" \
    --container-mounts "$PWD":/workspace,/shared/datasets/rsdo:/data \
    --container-entrypoint /workspace/bin/exec-pred.sh --run-path "./data/runs/run_2021-02-16T11:43:57"

echo "$SLURM_JOB_ID -> Done."

#wait
