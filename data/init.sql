CREATE TABLE public.data
(
    "0"      FLOAT,
    "1"      FLOAT,
    "2"      FLOAT,
    "3"      FLOAT,
    "4"      FLOAT,
    "5"      FLOAT,
    "6"      FLOAT,
    "7"      FLOAT,
    "8"      FLOAT,
    "9"      FLOAT,
    "target" FLOAT
);

COPY public.data(
    "0"     ,
    "1"     ,
    "2"     ,
    "3"     ,
    "4"     ,
    "5"     ,
    "6"     ,
    "7"     ,
    "8"     ,
    "9"     ,
    "target"
) FROM '/data/data_file.csv' DELIMITER ',' CSV HEADER;


CREATE TABLE public.models
(
    "model_name"       TEXT    NOT NULL,
    "model_weights"    BYTEA
);