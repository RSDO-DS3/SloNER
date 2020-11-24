#!/bin/bash

set -euo pipefail

DIRS=`ls -d $PWD/data/ne/gigafida/*/`

for a in $DIRS;
do
  files=`ls $a*.csv`
  for f in $files
  do
    psql -d gigafida -c "COPY rsdo.named_entities FROM '$f' delimiter ',' CSV HEADER;";
  done
done
