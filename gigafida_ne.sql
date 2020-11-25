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

CREATE MATERIALIZED VIEW rsdo.word_frequency
AS
SELECT f.lemma            "LEMMA",
       f.type             "TYPE",
       f.words[f.max_pos] "FORM",
       f.msds[f.max_pos]  "MSD",
       f.sum_freqs        "FREQ"
FROM (SELECT m.lemma,
             m.type,
             m.words,
             m.msds,
             m.sum_freqs,
             array_position(m.freqs, m.max_freqs) max_pos
      FROM (SELECT n.lemma,
                   n.type,
                   array_agg(n.word) words,
                   array_agg(n.msd)  msds,
                   array_agg(n.freq) freqs,
                   max(n.freq)       max_freqs,
                   SUM(n.freq)       sum_freqs
            FROM (SELECT word,
                         lemma,
                         msd,
                         type,
                         COUNT(*) freq
                  FROM rsdo.named_entities
                  GROUP BY lemma, word, msd, type
                  ORDER BY freq DESC) n
            GROUP BY n.lemma, n.type) m) f
ORDER BY sum_freqs DESC;

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
