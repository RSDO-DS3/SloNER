set -euo pipefail

REPO_PATH=/home/pr3mar/rsdo

for a in {00..99};
do
  psql -d gigafida -c "COPY rsdo.merged_named_entities FROM '$REPO_PATH/NER/data/ne/gigafida/GF$a-merged.csv' delimiter ',' CSV HEADER;";
done
