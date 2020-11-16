----------------------- Create the table
create table rsdo.named_entities
(
	word varchar,
	lemma varchar,
	msd varchar,
	type varchar
);

alter table rsdo.named_entities owner to postgres;

create index named_entities_word_index
	on rsdo.named_entities (word);

create index named_entities_lemma_index
	on rsdo.named_entities (lemma);

create index named_entities_type_index
	on rsdo.named_entities (type);

-- Insert data with bash one-liner:
-- for a in {00..99}; do psql -d gigafida -c "COPY rsdo.named_entities FROM '/path/to/repo>/NER/data/ne/gigafida/GF$a.csv' delimiter ',' CSV HEADER;"; done

-- Check the amount of inserted rows
SELECT COUNT(*) FROM rsdo.named_entities; -- 65 507 787

----------------------- Drop the views if they exist
DROP MATERIALIZED VIEW IF EXISTS word_frequency;
DROP MATERIALIZED VIEW IF EXISTS lemma_frequency;
DROP MATERIALIZED VIEW IF EXISTS type_frequency;

----------------------- word frequency
CREATE MATERIALIZED VIEW word_frequency
AS
SELECT
       word,
       substring(type from 3) as entity_type, -- trick to remove the B-/I- prefix from the type
       COUNT(*) frequency
FROM rsdo.named_entities
GROUP BY word, entity_type;

SELECT * FROM word_frequency ORDER BY frequency DESC LIMIT 10;

----------------------- lemma frequency
CREATE MATERIALIZED VIEW lemma_frequency
AS
SELECT
       lemma,
       substring(type from 3) as entity_type,
       COUNT(*) frequency
FROM rsdo.named_entities
GROUP BY lemma, entity_type;

SELECT * FROM lemma_frequency ORDER BY frequency, lemma ASC LIMIT 10;

----------------------- type frequency
CREATE MATERIALIZED VIEW type_frequency
AS
SELECT
       substring(type from 3) as entity_type,
       COUNT(*) frequency
FROM rsdo.named_entities
GROUP BY entity_type;

SELECT entity_type, frequency FROM type_frequency LIMIT 100;
