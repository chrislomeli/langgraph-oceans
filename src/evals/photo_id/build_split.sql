-- build_split.sql — freeze the Layer-A photo-ID eval split ONCE (the source of truth).
--
-- One held-out QUERY sighting per eligible individual (closed-set leave-one-out).
-- The pick is deterministic (md5 of sighting_id) so rebuilding on the SAME catalog
-- reproduces the SAME split forever — the fixed yardstick the baseline + the future
-- LoRA rerun both score against. This table is ALSO the train/test guardrail: Phase-C
-- LoRA must exclude these query sightings from training, or the lift is invalid.
--
-- The query sighting is left IN fluke_embeddings; "hold out" = exclude_sighting_id at
-- query time. The `role` column reserves the seam for an `open_set_novel` slice later.
--
-- Rebuild is safe + reproducible:  psql "$OCEANS_URL" -f build_split.sql

DROP TABLE IF EXISTS datasets.photo_id_eval_split;

CREATE TABLE datasets.photo_id_eval_split AS
WITH eligible AS (
    SELECT individual_id
    FROM datasets.fluke_embeddings
    WHERE embedder_ver = 'clip-vitb32-v1'
    GROUP BY individual_id
    HAVING count(*) >= 10                     -- eligibility: ≥9 gallery photos remain after holdout
),
ranked AS (
    SELECT fe.individual_id,
           fe.sighting_id,
           row_number() OVER (PARTITION BY fe.individual_id
                              ORDER BY md5(fe.sighting_id::text)) AS rn   -- deterministic, unbiased pick
    FROM datasets.fluke_embeddings fe
    JOIN eligible USING (individual_id)
    WHERE fe.embedder_ver = 'clip-vitb32-v1'
)
SELECT individual_id,
       sighting_id            AS query_sighting_id,
       'closed_set'::text     AS role
FROM ranked
WHERE rn = 1;

ALTER TABLE datasets.photo_id_eval_split
    ADD PRIMARY KEY (query_sighting_id);

COMMENT ON TABLE datasets.photo_id_eval_split IS
    'Frozen Layer-A photo-ID eval split: one held-out query sighting per eligible individual. Source of truth for the baseline AND the LoRA train/test separation.';
