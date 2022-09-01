#!/usr/bin/env bash

docker run \
	-d --rm \
	--name rsdo-ds3-ner-api-cpu \
	-p 5000:5000 \
	--env NER_MODEL_PATH="/app/data/models/bert-based/" \
	--mount type=bind,source="`pwd`/data/runs/run_2021-03-05T12:39:26/bert-base-multilingual-cased-bsnlp-2021-finetuned-5-epochs/",destination="/app/data/models/bert-based/",ro \
	rsdo-ds3-ner-api-cpu:v1
