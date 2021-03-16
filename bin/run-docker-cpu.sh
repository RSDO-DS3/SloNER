#!/usr/bin/env bash

docker run \
	--rm \
	--name ner-bert-cpu \
	--volume `pwd`/bin:/app/bin \
	--volume `pwd`/data:/app/data \
	--volume `pwd`/results:/app/results \
	--volume `pwd`/logs:/app/logs \
	--volume `pwd`/figures:/app/figures \
	rsdo-ds3-ner:v1-cpu \
		/app/bin/exec-bert.sh \
			--epochs 5 \
			--train \
			--test
