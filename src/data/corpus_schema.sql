-- corpus_schema.sql — the two NEW tables the tool/data design landed on.
-- Schema: datasets (same as fluke_embeddings, sightings, etc.). Idempotent.
-- Design: docs/design/tool-design.md + the per-collection design notes.
--
--   datasets.stock_status  — the SAR "card": a plain relational lookup, NO vector.
--   datasets.doc_chunks    — the RAG corpus: text + metadata + vector + keyword index.

-- ──────────────────────────────────────────────────────────────────────────
-- 1. stock_status — the SAR facts card (hand-curated, ~6 rows, no embedder)
--    Keyed on the species COMMON NAME → joins to individuals.common_name
--    (NOT individuals.species, which holds the scientific name).
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS datasets.stock_status (
    species                  text PRIMARY KEY,   -- common name, e.g. 'Humpback Whale' = individuals.common_name
    stock                    text NOT NULL,      -- primary stock, e.g. 'California/Oregon/Washington'
    abundance                integer,            -- best estimate
    abundance_cv             real,
    abundance_year           text,               -- e.g. '2015-2018'
    abundance_method         text,               -- e.g. 'mark-recapture'
    n_min                    integer,            -- minimum population estimate
    pbr                      real,               -- potential biological removal (full)
    pbr_us_waters            real,               -- PBR in U.S. waters (the jurisdictional subtlety)
    trend                    text,               -- e.g. '+8.2%/yr (increasing)'
    dominant_threat          text,               -- 'entanglement' | 'vessel strike'  (the F5 pivot)
    ship_strike_yr_observed  real,
    ship_strike_yr_estimated real,               -- the honest detection-gap pair
    entanglement_yr          real,
    total_human_mortality_yr real,
    exceeds_pbr              boolean,            -- → 'strategic' stock
    esa_status               text,
    mmpa_status              text,
    source                   text NOT NULL,      -- citation, e.g. 'Humpback CA/OR/WA SAR'
    year                     integer
);

-- ──────────────────────────────────────────────────────────────────────────
-- 2. doc_chunks — the RAG corpus (text + metadata join-keys + vector + tsv)
--    embedding: OpenAI text-embedding-3-large @ dimensions=1536 (under pgvector's
--    2000-dim index cap). species[] holds COMMON NAMES (matches individuals.common_name).
-- ──────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS datasets.doc_chunks (
    chunk_id      bigserial PRIMARY KEY,
    text          text NOT NULL,                 -- the chunk body (what gets cited)
    header        text,                          -- contextual header prepended before embedding
    -- metadata / join keys ────────────────────
    species       text[],                        -- common_names this chunk pertains to (NULL = species-agnostic, e.g. a mechanism review)
    sanctuary     text,                          -- NMS name (NULL for SAR / review)
    doc_type      text NOT NULL,                 -- 'SAR' | 'condition-report' | 'mgmt-plan' | 'review'
    section       text,                          -- source section heading, e.g. 'Vessel Strikes'
    stock         text,                          -- stock label (SAR) — keeps the 2-SAR species separable
    source        text NOT NULL,                 -- document / citation source
    year          integer,
    -- vectors ─────────────────────────────────
    embedding     vector(1536),                  -- openai-3-large @ 1536
    tsv           tsvector GENERATED ALWAYS AS    -- keyword half of hybrid search; auto-maintained
                  (to_tsvector('english', coalesce(header,'') || ' ' || coalesce(text,''))) STORED,
    embedder_ver  text NOT NULL                  -- e.g. 'openai-3-large-1536-v1'  (migration hook)
);

-- indexes: vector (HNSW cosine) + keyword (GIN) + the metadata pre-filter keys
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_hnsw
    ON datasets.doc_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS doc_chunks_tsv_gin
    ON datasets.doc_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS doc_chunks_species_gin
    ON datasets.doc_chunks USING gin (species);
CREATE INDEX IF NOT EXISTS doc_chunks_filter_btree
    ON datasets.doc_chunks (doc_type, sanctuary, embedder_ver);
