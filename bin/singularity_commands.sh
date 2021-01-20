#!/bin/bash

# build a new container:
singularity build <local_target>.sif docker://<URL>
# e.g.
singularity build ./containers/sing-pavlov-container.sif docker://pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# install dependencies within the container
singularity exec ./containers/sing-pavlov-container.sif pip install -r requirements.txt

# run the container
singularity run --nv ./containers/sing-container.sif

# submit a job to SLURM using a singularity container
sbatch -p compute --gres=gpu:1 --parsable ./bin/run-singularity-container.sh
