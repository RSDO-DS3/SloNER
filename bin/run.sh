#!/bin/bash
set -euo pipefail

SLURM_ACCOUNT=rsdo
SLURM_PARTITION=rsdo

echo "Running the BERT setup:"

SETUP=$(\
    sbatch \
        -A $SLURM_ACCOUNT \
        -p $SLURM_PARTITION \
        --parsable \
        ./bin/run-setup.sh
)

echo "The BERT setup ID is: $SETUP"

echo "Running the BERT training:"

TRAIN=$(\
    sbatch \
        -A $SLURM_ACCOUNT \
        -p $SLURM_PARTITION \
        --dependency=afterok:$SETUP \
        --parsable \
        ./bin/run-bert-train.sh
)

echo "The BERT training ID is: $TRAIN"

echo "Runnnig the BERT testing:"
TEST=$(\
    sbatch \
        -A $SLURM_ACCOUNT \
        -p $SLURM_PARTITION \
        --dependency=afterok:$TRAIN \
        --parsable \
        ./bin/run-bert-test.sh
)

echo "The BERT testing ID is: $TEST"
