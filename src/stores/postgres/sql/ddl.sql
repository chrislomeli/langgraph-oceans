create table happywhale
(
    image         varchar(30),
    species       varchar(60),
    individual_id varchar(60)
);

comment on table happywhale is 'Happywhale data that we are not using';

create table obis_seamap_points
(
    dataset_id              integer,
    row_id                  text,
    latitude                double precision,
    longitude               double precision,
    species_name            text,
    scientific_name         text,
    common_name             text,
    itis_tsn                integer,
    group_size              integer,
    series_id               text,
    date_time               text,
    timezone                text,
    ds_type                 text,
    platform                text,
    provider                text,
    lprecision              integer,
    tprecision              integer,
    oceano                  text,
    notes                   text,
    last_mod                text,
    type                    text,
    modified                text,
    language                text,
    institution_code        text,
    basis_of_record         text,
    occurrence_id           text,
    catalog_number          text,
    recorded_by             text,
    individual_count        text,
    sex                     text,
    preparations            text,
    occurrence_remarks      text,
    organism_id             text,
    organism_name           text,
    organism_remarks        text,
    event_id                text,
    event_date              text,
    event_time              text,
    higher_geography        text,
    water_body              text,
    locality                text,
    verbatim_locality       text,
    decimal_latitude        text,
    decimal_longitude       text,
    geodetic_datum          text,
    coordinate_precision    text,
    taxon_rank              text,
    vernacular_name         text,
    license                 text,
    rights_holder           text,
    external_resource       text,
    external_resource_thumb text,
    location                geography(Point, 4326),
    individual_id           bigint,
    individual_url          text
);

comment on table obis_seamap_points is 'This is a staging table of obis-seamap data for the north pacific, and feeds sightings and individuals tables';

create index obis_seamap_points__index
    on obis_seamap_points using gist (location);

create table ais_2024
(
    rid  serial
        primary key,
    rast raster
        constraint enforce_srid_rast
            check (st_srid(rast) = 3857)
        constraint enforce_scalex_rast
            check (round((st_scalex(rast))::numeric, 10) = round((100)::numeric, 10))
        constraint enforce_scaley_rast
            check (round((st_scaley(rast))::numeric, 10) = round((- (100)::numeric), 10))
        constraint enforce_width_rast
            check (st_width(rast) = ANY (ARRAY [256, 110]))
        constraint enforce_height_rast
            check (st_height(rast) = ANY (ARRAY [256, 181]))
        constraint enforce_same_alignment_rast
            check (st_samealignment(rast,
                                    '0100000000000000000000594000000000000059C0E4C798B754F26BC155302A9955A4594100000000000000000000000000000000110F000001000100'::raster))
        constraint enforce_num_bands_rast
            check (st_numbands(rast) = 1)
        constraint enforce_pixel_types_rast
            check (_raster_constraint_pixel_types(rast) = '{32BUI}'::text[])
        constraint enforce_nodata_values_rast
            check (_raster_constraint_nodata_values(rast) = '{0.0000000000}'::numeric[])
        constraint enforce_out_db_rast
            check (_raster_constraint_out_db(rast) = '{f}'::boolean[])
        constraint enforce_max_extent_rast
            check (st_envelope(rast) @
                   '0103000020110F00000100000005000000F263CC5BAA6A72C154C1A864A2C63741F263CC5BAA6A72C155302A9955A45941C88F316FEB7A59C155302A9955A45941C88F316FEB7A59C154C1A864A2C63741F263CC5BAA6A72C154C1A864A2C63741'::geometry)
);

comment on table ais_2024 is 'AIS shipping lanes - 2024';

create index ais_2024_st_convexhull_idx
    on ais_2024 using gist (st_convexhull(rast));

create table ais_2025
(
    rid  serial
        primary key,
    rast raster
        constraint enforce_srid_rast
            check (st_srid(rast) = 3857)
        constraint enforce_scalex_rast
            check (round((st_scalex(rast))::numeric, 10) = round((100)::numeric, 10))
        constraint enforce_scaley_rast
            check (round((st_scaley(rast))::numeric, 10) = round((- (100)::numeric), 10))
        constraint enforce_width_rast
            check (st_width(rast) = ANY (ARRAY [256, 228]))
        constraint enforce_height_rast
            check (st_height(rast) = ANY (ARRAY [256, 222]))
        constraint enforce_same_alignment_rast
            check (st_samealignment(rast,
                                    '0100000000000000000000594000000000000059C0E4C798B7D80E6CC155302A9955A4594100000000000000000000000000000000110F000001000100'::raster))
        constraint enforce_num_bands_rast
            check (st_numbands(rast) = 1)
        constraint enforce_pixel_types_rast
            check (_raster_constraint_pixel_types(rast) = '{32BUI}'::text[])
        constraint enforce_nodata_values_rast
            check (_raster_constraint_nodata_values(rast) = '{0.0000000000}'::numeric[])
        constraint enforce_out_db_rast
            check (_raster_constraint_out_db(rast) = '{f}'::boolean[])
        constraint enforce_max_extent_rast
            check (st_envelope(rast) @
                   '0103000020110F00000100000005000000F263CC5B6C3A72C154C1A8649E123141F263CC5B6C3A72C155302A9955A459411C38674849D96F4155302A9955A459411C38674849D96F4154C1A8649E123141F263CC5B6C3A72C154C1A8649E123141'::geometry)
);

comment on table ais_2025 is 'AIS shipping lanes - 2025';

create index ais_2025_st_convexhull_idx
    on ais_2025 using gist (st_convexhull(rast));

create table ais_2023
(
    rid  serial
        primary key,
    rast raster
        constraint enforce_srid_rast
            check (st_srid(rast) = 3857)
        constraint enforce_scalex_rast
            check (round((st_scalex(rast))::numeric, 10) = round((100)::numeric, 10))
        constraint enforce_scaley_rast
            check (round((st_scaley(rast))::numeric, 10) = round((- (100)::numeric), 10))
        constraint enforce_width_rast
            check (st_width(rast) = ANY (ARRAY [256, 110]))
        constraint enforce_height_rast
            check (st_height(rast) = 256)
        constraint enforce_same_alignment_rast
            check (st_samealignment(rast,
                                    '0100000000000000000000594000000000000059C08C5C1C19D5FE6BC134430D0D48A4594100000000000000000000000000000000110F000001000100'::raster))
        constraint enforce_num_bands_rast
            check (st_numbands(rast) = 1)
        constraint enforce_pixel_types_rast
            check (_raster_constraint_pixel_types(rast) = '{32BUI}'::text[])
        constraint enforce_nodata_values_rast
            check (_raster_constraint_nodata_values(rast) = '{0.0000000000}'::numeric[])
        constraint enforce_out_db_rast
            check (_raster_constraint_out_db(rast) = '{f}'::boolean[])
        constraint enforce_max_extent_rast
            check (st_envelope(rast) @
                   '0103000020110F00000100000005000000462E8E8CAA6A72C1D00C3534200D3841462E8E8CAA6A72C134430D0D48A4594118B93832EC7A59C134430D0D48A4594118B93832EC7A59C1D00C3534200D3841462E8E8CAA6A72C1D00C3534200D3841'::geometry)
);

comment on table ais_2023 is 'AIS shipping lanes - 2023';

create index ais_2023_st_convexhull_idx
    on ais_2023 using gist (st_convexhull(rast));

create table ais_2022
(
    rid  serial
        primary key,
    rast raster
        constraint enforce_srid_rast
            check (st_srid(rast) = 3857)
        constraint enforce_scalex_rast
            check (round((st_scalex(rast))::numeric, 10) = round((100)::numeric, 10))
        constraint enforce_scaley_rast
            check (round((st_scaley(rast))::numeric, 10) = round((- (100)::numeric), 10))
        constraint enforce_width_rast
            check (st_width(rast) = ANY (ARRAY [256, 110]))
        constraint enforce_height_rast
            check (st_height(rast) = 256)
        constraint enforce_same_alignment_rast
            check (st_samealignment(rast,
                                    '0100000000000000000000594000000000000059C0E4C798B7540B6CC155302A9955A4594100000000000000000000000000000000110F000001000100'::raster))
        constraint enforce_num_bands_rast
            check (st_numbands(rast) = 1)
        constraint enforce_pixel_types_rast
            check (_raster_constraint_pixel_types(rast) = '{32BUI}'::text[])
        constraint enforce_nodata_values_rast
            check (_raster_constraint_nodata_values(rast) = '{0.0000000000}'::numeric[])
        constraint enforce_out_db_rast
            check (_raster_constraint_out_db(rast) = '{f}'::boolean[])
        constraint enforce_max_extent_rast
            check (st_envelope(rast) @
                   '0103000020110F00000100000005000000F263CC5B2A4572C154C1A864560D3841F263CC5B2A4572C155302A9955A45941C88F316FEB7A59C155302A9955A45941C88F316FEB7A59C154C1A864560D3841F263CC5B2A4572C154C1A864560D3841'::geometry)
);

comment on table ais_2022 is 'AIS shipping lanes 2022';

create index ais_2022_st_convexhull_idx
    on ais_2022 using gist (st_convexhull(rast));

create table individuals
(
    individual_url text   not null,
    species        text,
    n_sightings    integer,
    first_seen     date,
    last_seen      date,
    individual_id  bigint not null
        constraint individuals_pk
            primary key
);

comment on table individuals is 'Individual whale ids - parent of sightings table';

create table sightings
(
    sighting_id    bigserial
        primary key,
    individual_url text not null,
    species        text,
    obs_date       date,
    location       geography(Point, 4326),
    group_size     integer,
    image_url      text,
    thumb_url      text,
    license        text,
    rights_holder  text,
    provider       text,
    source_row_id  text,
    individual_id  bigint
        constraint sightings_individuals_individual_id_fk
            references individuals
);

comment on table sightings is 'Whale sightings north pacific - clean set';

create index sightings_location_idx
    on sightings using gist (location);

create index sightings_individual_id_index
    on sightings (individual_id);

create table manifest
(
    sighting_id   integer,
    individual_id integer,
    asset_ref     text,
    source_url    text,
    status        text
);

comment on table manifest is 'working table - was used to load images - keep or drop';

create table fluke_embeddings
(
    image_id      bigserial
        primary key,
    individual_id bigint      not null,
    sighting_id   bigint      not null
        constraint fk_sighting
            references sightings,
    asset_ref     text        not null,
    source_url    text,
    embedding     vector(512) not null,
    embedder_ver  text        not null,
    created_at    timestamp with time zone default now(),
    constraint uq_sighting_ver
        unique (sighting_id, embedder_ver)
);

comment on table fluke_embeddings is 'Contains embedded ''images'' of whales for saerch';

create index fluke_embeddings_embedding_idx
    on fluke_embeddings using hnsw (embedding public.vector_cosine_ops);

create or replace view ais_all(year, rast) as
SELECT 2022 AS year,
       ais_2022.rast
FROM ais_2022
UNION ALL
SELECT 2023 AS year,
       ais_2023.rast
FROM ais_2023
UNION ALL
SELECT 2024 AS year,
       ais_2024.rast
FROM ais_2024
UNION ALL
SELECT 2025 AS year,
       ais_2025.rast
FROM ais_2025;

create table if not exists doc_chunks
(
    chunk_id     text not null
        primary key,
    text         text not null,
    header       text,
    species      text[],
    sanctuary    text,
    doc_type     text not null,
    section      text,
    stock        text,
    source       text not null,
    year         integer,
    embedding    vector(1536),
    tsv          tsvector generated always as (to_tsvector('english'::regconfig,
                                                           ((COALESCE(header, ''::text) || ' '::text) ||
                                                            COALESCE(text, ''::text)))) stored,
    embedder_ver text not null
);

create index if not exists doc_chunks_embedding_hnsw
    on doc_chunks using hnsw (embedding public.vector_cosine_ops);

create index if not exists doc_chunks_tsv_gin
    on doc_chunks using gin (tsv);

create index if not exists doc_chunks_species_gin
    on doc_chunks using gin (species);

create index if not exists doc_chunks_filter_btree
    on doc_chunks (doc_type, sanctuary, embedder_ver);
