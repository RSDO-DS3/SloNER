#!/bin/bash
# TODO: add sbatch specifics
set -euo pipefail


# the following command opens a bash terminal of an already existing container
# with the current directory (.) mounted
srun \
--gpus=1 \
--container-image ./pytorch-image.sqfs \
--container-save pytorch-image.sqfs \
--container-mount-home \
--container-remap-root \
--container-mounts .:/workspace \
--pty bash -l
