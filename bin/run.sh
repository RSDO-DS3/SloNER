#!/bin/bash
set -euo pipefail

SLURM_ACCOUNT=rsdo
SLURM_PARTITION=rsdo

echo "Running the setup:"

SETUP=$(\
    sbatch \
        -A $SLURM_ACCOUNT \
        -p $SLURM_PARTITION \
        --parsable \
        ./bin/run-setup.sh
)

echo "Setup id is $SETUP"

echo "Running the train:"

TRAIN=$(\
    sbatch \
        -A $SLURM_ACCOUNT \
        -p $SLURM_PARTITION \
        --dependency=afterok:$SETUP \
        --parsable \
        ./bin/run-container.sh
)

echo "Train id is: $TRAIN"
