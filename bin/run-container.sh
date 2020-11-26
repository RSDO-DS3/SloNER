#!/bin/bash

CONTAINER_IMAGE_PATH="$PWD/containers/pytorch-image.sqfs"

srun \
	--gpus=1\
	--container-image "$CONTAINER_IMAGE_PATH" \
	--container-save "$CONTAINER_IMAGE_PATH" \
	--container-mounts .:/workspace \
	--pty bash -l
