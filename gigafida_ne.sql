create schema rsdo;

create table rsdo.named_entities
(
    word  varchar,
    lemma varchar,
    msd   varchar,
    type  varchar
);

alter table rsdo.named_entities
    owner to postgres;

create index named_entities_word_index
    on rsdo.named_entities (word);

create index named_entities_lemma_index
    on rsdo.named_entities (lemma);

create index named_entities_type_index
    on rsdo.named_entities (type);

SELECT * FROM rsdo.named_entities;

CREATE MATERIALIZED VIEW word_frequency
AS
SELECT
       nn.words[1] "FORM",
       nn.type "TYPE",
       nn.freqs "FREQ",
       nn.lemma "LEMMA",
       nn.msds[1] "MSD"
FROM (SELECT n.lemma,
             n.type,
             array_agg(n.word) words,
             array_agg(n.msd)  msds,
             SUM(n.freq)       freqs
      FROM (SELECT word,
                   lemma,
                   msd,
                   type,
                   COUNT(*) freq
            FROM rsdo.named_entities
            GROUP BY lemma, word, msd, type
            ORDER BY freq DESC) n
      GROUP BY n.lemma, n.type
      ORDER BY freqs DESC) nn;

-- final result
SELECT row_number() OVER (ORDER BY 1) id,
       m."FORM",
       m."TYPE",
       m."FREQ",
       m."LEMMA",
       replace(m."MSD", ' ', ' ')     "MSD",
       split_part(m."LEMMA", ' ', 1)  "LEMMA.1",
       split_part(m."MSD", ' ', 1)    "MSD.1",
       split_part(m."LEMMA", ' ', 2)  "LEMMA.2",
       split_part(m."MSD", ' ', 2)    "MSD.2",
       split_part(m."LEMMA", ' ', 3)  "LEMMA.3",
       split_part(m."MSD", ' ', 3)    "MSD.3",
       split_part(m."LEMMA", ' ', 4)  "LEMMA.4",
       split_part(m."MSD", ' ', 4)    "MSD.4",
       split_part(m."LEMMA", ' ', 5)  "LEMMA.5",
       split_part(m."MSD", ' ', 5)    "MSD.5",
       split_part(m."LEMMA", ' ', 6)  "LEMMA.6",
       split_part(m."MSD", ' ', 6)    "MSD.6"
FROM word_frequency m
ORDER BY m."FREQ" DESC;
