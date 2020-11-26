#!/bin/bash

srun \
	--gpus=1\
       	--container-image ./containers/pytorch-image.sqfs \
	--container-mounts .:/workspace \
	--pty bash -l
