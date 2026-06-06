# Photo-ID (the ML piece) — schema + download/embed plan

> Goal: turn a fluke **photo** into an **`organism_id`** (or "not in catalog"). This
> is the perception front-end; its output is the exact key the database hub already
> uses. The two halves meet at `organism_id`:
>
> ```
> [photo] → PHOTO-ID MODEL → organism_id → individuals / sightings / range / AIS / zones
>             (this doc)                       (the database, already built)
> ```

Status: planning, 2026-06-05. Depends on the completed data layer (`data-build-plan.md`).

---

## 0. The mental model (read once)

Photo-ID is **embedding + nearest-neighbor search**, not classification:
1. A model turns each image into a **vector** ("embedding") such that photos of the
   *same* whale land close together, *different* whales far apart.
2. The **catalog** = all known whales' images, pre-embedded into vectors (in pgvector).
3. To identify a new photo: embed it, find the **nearest** catalog vectors → those
   are the candidate individuals. If nothing is close enough → **abstain** ("novel").

So the images live on **disk**, the **vectors** live in Postgres (pgvector), and the
agent's `photo_id` tool is a nearest-neighbor query with a confidence threshold.

---

## 1. Schema — `fluke_embeddings` (the vectors live here)

Images are **not** stored in Postgres (binary blobs belong on disk). Postgres stores
one **vector per image**, keyed to its individual.

```sql
CREATE TABLE datasets.fluke_embeddings (
  image_id      bigserial PRIMARY KEY,
  individual_id text NOT NULL REFERENCES datasets.individuals(individual_id),
  sighting_id   bigint REFERENCES datasets.sightings(sighting_id),  -- which encounter this photo is from
  asset_ref     text NOT NULL,           -- local path to the downloaded JPEG (the bytes live here)
  source_url    text,                    -- the external_resource URL it came from
  license       text,                    -- carried for attribution
  embedding     vector(512),             -- the fluke vector — DIM MUST MATCH YOUR MODEL (see §3)
  embedder_ver  text NOT NULL,           -- e.g. 'clip-vitb32-v1' — lets you re-embed without a wipe
  created_at    timestamptz DEFAULT now()
);

-- fast cosine nearest-neighbor search:
CREATE INDEX ON datasets.fluke_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON datasets.fluke_embeddings (individual_id);
CREATE INDEX ON datasets.fluke_embeddings (embedder_ver);
```

Why each unusual field:
- **`embedder_ver`** — when you swap/retrain the model, the new vectors are a *different
  space* and can't be compared to old ones. You re-embed into new rows tagged with a new
  version and switch over with a `WHERE embedder_ver = …` filter. (Migration = a filter,
  not a wipe — the same lossless habit.)
- **`asset_ref`** — the bytes are on disk; the DB just points at them.
- **One row per image**, many images per individual → that's the catalog richness.

---

## 2. Download the images (`external_resource` → disk)

Pull only what you'll use, politely. Filters: has a photo, open license, **exclude
no-derivatives** (embedding is arguably a derivative — drop the 33 `by-nc-nd` rows).

```sql
-- the download manifest (scope to your demo region to keep it small)
SELECT sighting_id, individual_id, image_url, license
FROM datasets.sightings
WHERE image_url IS NOT NULL
  AND license NOT LIKE '%-nd/%'                 -- exclude no-derivatives (ML safety)
  AND latitude BETWEEN 32 AND 42                -- demo region (drop for full set)
ORDER BY individual_id;
```

Then a small **Python** downloader (this is the learning bit — keep it dumb and idempotent):
```
for each (sighting_id, individual_id, image_url):
    path = blob_dir / safe(individual_id) / f"{sighting_id}.jpg"
    if path exists: skip                          # idempotent — re-runnable
    GET image_url  (timeout, retry once)          # these are public CC/CC0 CDN files
    save to path
    remember (sighting_id, individual_id, path, image_url, license)
```
- Use the **medium** URL (`…-m.happywhale.com/…jpg`, ~117 KB) for ID quality; thumbs
  (`-t`, ~6 KB) only if you want the whole coast cheaply.
- Be a good citizen: modest concurrency (e.g. 4–8), a small delay, one retry. It's
  open data served for this purpose, but don't hammer it.
- Carry `license` + `provider`/`rights_holder` (from `sightings`) for attribution.

Scale check: a region subset is a few thousand–tens-of-thousands of images
(manageable). The full ~110k mediums ≈ 13 GB — only if you really want it.

---

## 3. Embed the images (the model)

Two depths — **start with the baseline, measure, then improve.**

**Baseline (get the whole pipeline working, no training):**
- Use a pretrained image embedder — easiest is **OpenCLIP** (`open_clip`) ViT-B/32
  (→ `vector(512)`) or ViT-L/14 (→ `vector(768)`; set the column dim to match).
- For each downloaded image: load → preprocess → model → L2-normalize → a vector.
- Insert into `fluke_embeddings` with `embedder_ver='clip-vitb32-v1'`.
- ⚠ Honest caveat: CLIP is trained on web images, not fine-grained flukes — it'll get
  the *pipeline* working and give a baseline number, but expect modest accuracy. That's
  fine; the point is to have something end-to-end to measure.

**The performance step (the personal LoRA goal):**
- Fine-tune a backbone with **metric learning** (ArcFace or triplet loss) so same-individual
  flukes cluster — this is what the Happywhale Kaggle winners did. Apply it via **LoRA**
  adapters (small, fast, the bounded "fun ML").
- Re-embed everything with `embedder_ver='lora-arcface-v1'`, re-measure, show the lift
  over the CLIP baseline.
- *(This is the one place training happens; everything else stays off-the-shelf.)*

Tools: `torch` + `open_clip`/`timm` for the model, `pgvector` Python adapter + `psycopg`
to write vectors. GPU helps but CPU works for a few thousand images.

---

## 4. Query = the `photo_id` tool (nearest-neighbor + abstain)

```sql
-- given an embedded query vector :q (same model/version), find candidate individuals
SELECT individual_id,
       max(1 - (embedding <=> :q)) AS score,   -- cosine similarity; best image per whale
       count(*)                    AS n_catalog_images
FROM datasets.fluke_embeddings
WHERE embedder_ver = 'clip-vitb32-v1'
GROUP BY individual_id
ORDER BY score DESC
LIMIT 5;
```
- Aggregate by individual (a whale has many images — take its **best** match).
- **Abstain** if the top `score` < a calibrated threshold → output `NOVEL` instead of a
  fabricated ID. This is the open-set behavior; the agent decides what to do with it.

The tool returns: ranked `individual_id`s + scores + an `abstain` flag — *typed, dumb*,
exactly the contract the agent layer expects.

---

## 5. Evaluate (how you know it works)

Hold out data, then score — reuses the project's retrieval scorer:
- **Closed-set** (the whale *is* in the catalog): hold out some images, query them,
  measure **Precision@k / MRR** (does the right individual rank near the top?).
- **Open-set** (the whale is *not* in the catalog): hold out **entire individuals**,
  query them → the model should **abstain**. Measure: % correctly called `NOVEL` vs.
  fabricated a match. *This is the crown-jewel metric* — "it doesn't hallucinate identity."
- Your **9,624 seen-once** whales are the natural hard cases / open-set tail.

The threshold for abstain is calibrated on a held-out set (real work — raw cosine isn't a
probability).

---

## 6. Start here (sequence — components first)

1. **Download ~a few hundred images** from one small area. Just prove the downloader works.
2. **Embed them with CLIP** → write to `fluke_embeddings` (`embedder_ver='clip-vitb32-v1'`).
3. **Run the §4 query** with one held-out image → watch it return ranked individuals.
   *(That's the whole loop working — celebrate here.)*
4. **Scale to the demo region**, add the **held-out eval** (§5) → get a baseline number.
5. **Then** LoRA metric-learning fine-tune → re-embed (`v2`) → re-measure → show the lift.

Don't build the agent tool wrapper until the query + eval work standalone (feed it from
cases, not the live agent) — same components-first discipline as the data layer.

---

## Open questions to decide as you go
- **Model/dim**: CLIP ViT-B/32 (512) to start; revisit for the fine-tuned backbone.
- **Image size**: medium for the catalog; confirm it's enough resolution for flukes.
- **One vector per image** (above) vs. one averaged vector per individual — start per-image.
- **Kaggle**: optional extra training data + leaderboard benchmark; the catalog is the
  OBIS-Happywhale images (real `organism_id`s). Don't try to join the two namespaces.
