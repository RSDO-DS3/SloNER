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

create table rsdo.merged_named_entities
(
	word varchar,
	lemma varchar,
	msd varchar,
	entity_type varchar
);

alter table rsdo.merged_named_entities owner to postgres;

create index merged_named_entities_word_index
	on rsdo.merged_named_entities (word);

create index merged_named_entities_lemma_index
	on rsdo.merged_named_entities (lemma);

create index merged_named_entities_type_index
	on rsdo.merged_named_entities (entity_type);

-- Insert data with bash one-liner:
-- for a in {00..99}; do psql -d gigafida -c "COPY rsdo.named_entities FROM '/path/to/repo>/NER/data/ne/gigafida/GF$a.csv' delimiter ',' CSV HEADER;"; done

-- Check the amount of inserted rows
SELECT COUNT(*) FROM rsdo.named_entities; -- 65 507 787

----------------------- Drop the views if they exist
DROP MATERIALIZED VIEW IF EXISTS word_frequency;
DROP MATERIALIZED VIEW IF EXISTS lemma_frequency;
DROP MATERIALIZED VIEW IF EXISTS type_frequency;

----------------------- word frequency
SELECT * FROM word_frequency ORDER BY frequency DESC LIMIT 10;
CREATE MATERIALIZED VIEW m_word_frequency
AS
SELECT word "FORM",
       lemma "LEMMA",
       entity_type "TYPE",
       msd "MSD",
       COUNT(*) "FREQ"
FROM rsdo.merged_named_entities
GROUP BY word, lemma, msd, entity_type;


----------------------- get frequency the list
SELECT
    row_number() OVER (ORDER BY 1) id,
    m."FORM",
    m."LEMMA",
    m."TYPE",
    replace(m."MSD", ';', ' ') "MSD",
    m."FREQ",
    split_part(m."LEMMA", ' ', 1) "LEMMA.1",
    split_part(m."MSD", ';', 1) "MSD.1",
    split_part(m."LEMMA", ' ', 2) "LEMMA.2",
    split_part(m."MSD", ';', 2) "MSD.2",
    split_part(m."LEMMA", ' ', 3) "LEMMA.3",
    split_part(m."MSD", ';', 3) "MSD.3",
    split_part(m."LEMMA", ' ', 4) "LEMMA.4",
    split_part(m."MSD", ';', 4) "MSD.4",
    split_part(m."LEMMA", ' ', 5) "LEMMA.5",
    split_part(m."MSD", ';', 5) "MSD.5",
    split_part(m."LEMMA", ' ', 6) "LEMMA.6",
    split_part(m."MSD", ';', 6) "MSD.6"
FROM m_word_frequency m
-- WHERE m."TYPE" = 'per' -- can also be 'loc', 'org', 'misc', and 'deriv-per'
ORDER BY m."FREQ" DESC;

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
