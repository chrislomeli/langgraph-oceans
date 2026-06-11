-- build_reid_split.sql — freeze the Phase-C train/val/test split ONCE (source of truth).
--
-- This is the GENERALIZATION split for LoRA: it partitions WHOLE INDIVIDUALS into
-- train / val / test so that a test whale was NEVER seen during training. That is
-- the honest measure of re-ID — "can the embedding re-identify a whale it never
-- trained on?" — not just "can it re-spot a whale it memorized." Contrast with
-- build_split.sql, which is a closed-set, by-SIGHTING leave-one-out over ALL whales.
--
-- Choices baked in (decided in the design discussion):
--   * Split by individual (disjoint) — assigning a whole whale to one split means
--     all its photos follow, so there is zero identity leakage by construction.
--   * Deterministic hash of individual_id (md5 → bucket) — reproducible forever on
--     the same catalog, so the frozen-CLIP baseline and the LoRA rerun score the
--     SAME test whales. md5 is stable across Postgres versions (hashtext is not).
--   * Ratios ~70 / 15 / 15 (train / val / test), among eval-eligible whales.
--   * Eligibility floors:
--       - TRAIN needs >=2 photos  (>=1 positive pair to learn from)
--       - VAL/TEST need >=4 photos (a meaningful gallery remains after we hold out
--         one query photo). Whales with 2-3 photos can't anchor a fair eval, so they
--         fall through to TRAIN (used as training data / distractors, never as eval).
--   * One deterministic held-out QUERY sighting per val/test whale; that whale's
--     OTHER photos are its gallery. TRAIN whales have no held-out query (null).
--
-- GALLERY SCOPE (enforced at EVAL time, not here): a test query is matched only
-- against OTHER TEST whales — the clean disjoint protocol. The eval's search filters
-- the candidate pool to `individual_id IN (SELECT individual_id FROM reid_split
-- WHERE split='test')` and excludes the query's own sighting. This table just labels
-- the whales and picks the queries; the WHERE clause does the scoping.
--
-- Rebuild is safe + reproducible:  psql "$OCEANS_URL" -f build_reid_split.sql

DROP TABLE IF EXISTS datasets.reid_split;

CREATE TABLE datasets.reid_split AS
WITH eligible AS (                       -- the universe: whales with at least one pair
    SELECT individual_id, count(*) AS n_photos
    FROM datasets.fluke_embeddings
    WHERE embedder_ver = 'clip-vitb32-v1'      -- enumerate on the baseline catalog;
    GROUP BY individual_id                     -- the split is sighting/individual IDs,
    HAVING count(*) >= 2                        -- which are model-independent
),
hashed AS (                              -- deterministic 0..99 bucket per individual
    SELECT
        individual_id,
        n_photos,
        -- 'x'||<6 hex> → 24-bit int (a well-known md5→int idiom); mod 100 → bucket
        mod(('x' || left(md5(individual_id::text), 6))::bit(24)::int, 100) AS bucket
    FROM eligible
),
assigned AS (                            -- bucket + eligibility floor → split label
    SELECT
        individual_id,
        n_photos,
        CASE
            WHEN n_photos < 4 THEN 'train'   -- too thin for a fair gallery → training only
            WHEN bucket >= 85 THEN 'test'    -- top 15 buckets
            WHEN bucket >= 70 THEN 'val'     -- next 15 buckets
            ELSE 'train'                     -- bottom 70 buckets
        END AS split
    FROM hashed
),
query_pick AS (                          -- one held-out query sighting per eval whale
    SELECT individual_id, query_sighting_id
    FROM (
        SELECT
            fe.individual_id,
            fe.sighting_id AS query_sighting_id,
            row_number() OVER (PARTITION BY fe.individual_id
                               ORDER BY md5(fe.sighting_id::text)) AS rn  -- deterministic pick
        FROM datasets.fluke_embeddings fe
        JOIN assigned a USING (individual_id)
        WHERE fe.embedder_ver = 'clip-vitb32-v1'
          AND a.split IN ('val', 'test')       -- train whales keep all photos (no query)
    ) ranked
    WHERE rn = 1
)
SELECT
    a.individual_id,
    a.split,
    a.n_photos,
    q.query_sighting_id                  -- NULL for train rows
FROM assigned a
LEFT JOIN query_pick q USING (individual_id);

ALTER TABLE datasets.reid_split
    ADD PRIMARY KEY (individual_id);

CREATE INDEX reid_split_split_idx ON datasets.reid_split (split);

COMMENT ON TABLE datasets.reid_split IS
    'Frozen Phase-C train/val/test split, disjoint BY INDIVIDUAL (generalization protocol). '
    'train>=2 photos; val/test>=4 with one deterministic held-out query_sighting_id each. '
    'Source of truth for LoRA training data AND the by-individual re-ID eval.';

-- ── sanity check (run after building) ────────────────────────────────────────────
SELECT split, count(*) AS individuals, sum(n_photos) AS images,
       round(avg(n_photos), 1) AS mean_photos,
       count(query_sighting_id) AS held_out_queries
FROM datasets.reid_split GROUP BY split ORDER BY split;

-- Expect: train has 0 held_out_queries; val/test have one per individual; mean_photos
-- similar across splits (the hash is independent of photo count → stratified for free).
