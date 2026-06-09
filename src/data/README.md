# Whale Photo-ID Data Pipeline — Plain-English Guide

A walkthrough of what we're building, the ideas behind it, the programs, and
every trap we fell into (and why). Read top to bottom; nothing assumed.

---

## 1. The goal, in one sentence

Hand the system a photo of a whale, and it answers **"this is whale #4606"** —
or honestly says **"I've never seen this whale before"** instead of guessing.
That's a *photo-ID system*.

---

## 2. The big picture (the whole pipeline)

```
sightings table ──► DOWNLOAD ──► EMBED ──► fluke_embeddings ──► QUERY ──► EVAL
  (the data)        (images       (turn imgs   (the vector DB)    (search)   (how good
                     to disk)      into numbers)                              is it?)
```

Each box is a step. We've done the first three.

---

## 3. The ideas you need (no jargon)

### What is an "embedding"?
A trained AI model (here, **CLIP**) takes an image and spits out a fixed list of
numbers — **512 of them**. Think of those numbers as **GPS coordinates in a
512-dimensional space**. The magic property: **images that look alike land close
together**. Two photos of the same whale's tail → two nearby points. A different
whale → a far-away point.

- We are **not training** the model. We just feed it images and collect the
  numbers it hands back ("inference").
- A vector by itself is **meaningless** — 512 numbers say nothing. It's only
  useful because we store the whale's ID *next to* it. **That link is the whole
  product.**

### What is "CLIP ViT-B/32"? What is "512"?
- **CLIP** = the model that turns images into vectors.
- **ViT-B/32** = the specific version of CLIP. (ViT = Vision Transformer; B =
  "Base" size; /32 = how finely it chops the image.)
- Each model version outputs a **fixed-length** vector. ViT-B/32 → **512**
  numbers. A bigger model (ViT-L/14) → 768. **The database column width must
  match the model's output** (`vector(512)`), or inserts fail.

### What is "cosine similarity"?
The way we measure "how close are two vectors." Closer = more similar = more
likely the same whale. We "L2-normalize" each vector (scale it to length 1) so
this comparison is clean.

### What is "embedder_ver"? (versioning)
Every vector is stamped with the model that made it, e.g. `clip-vitb32-v1`.
Why? **Vectors from different models are not comparable** — like comparing
inches to centimeters without knowing which is which. If we later train a better
model, we re-embed everything with a new stamp (`...-v2`). Old and new vectors
**coexist**; we just filter by version. **Upgrading = adding rows, never a wipe.**

### What is "idempotent"?
A program you can run again and again safely — re-running does no harm and
creates no duplicates. Critical here because embedding 114K images takes a long
time and *will* get interrupted. Our embed job checks "do I already have a vector
for this image at this version?" and skips it. So: kill it, rerun it, no problem.

---

## 4. The data we started with

- A Postgres database called **`oceans`**, schema **`datasets`**.
- **`sightings`** table — each row is one whale encounter: `sighting_id`,
  `individual_id`, `image_url`, `license`, location, date. ~114K rows.
- The images themselves live on a public server (Happywhale), one URL per row.

---

## 5. The programs (in order)

### A. `download_images.py` — DONE
Reads each sighting's `image_url`, downloads the JPEG, saves it to disk at
`src/data/blobs/<individual_id>/<sighting_id>.jpg`, and records what it did in a
**`manifest`** table: `(sighting_id, individual_id, asset_ref, source_url,
status)`.

- **Why a separate "manifest"?** It's the bridge between "files on disk" and
  "what to embed next." The file *path itself* encodes the whale's identity.
- Used a **thread pool** (downloading is slow *waiting on the network*, so doing
  many at once helps). Idempotent: skip files already on disk.

### B. `embed_images.py` — DONE (running now)
The big one. For every manifest row:
1. Load the JPEG from disk.
2. Preprocess it (resize/normalize the way CLIP expects).
3. Run it through CLIP → a 512-number vector.
4. L2-normalize.
5. Insert into **`fluke_embeddings`** with the whale's `individual_id` /
   `sighting_id` and the `embedder_ver` stamp.

- Uses **batching** (32 images per model call), *not* threads — embedding is
  math-heavy (CPU/GPU), so batching is the right lever, not concurrency.
- Idempotent via a `NOT EXISTS` check + a database `UNIQUE` constraint.
- Runs on the Mac GPU ("MPS"). Takes tens of minutes to a couple hours.

### C. `photo_id.py` — NEXT (the payoff)
Takes one query photo → embeds it the same way → asks the vector DB "which
stored vectors are nearest?" → returns ranked `individual_id`s + a confidence
score + an **abstain** flag (says "NOVEL" if nothing is close enough, instead of
fabricating an ID). Small program, big moment — this is where it first *works*.

### D. `eval.py` — LATER
Hide some images, query them, measure whether the right whale ranks at the top.
Gives an accuracy number. (And the crown-jewel test: does it correctly *abstain*
on whales it's never seen?) We don't build this until C works.

*(Way later: optionally fine-tune the model with LoRA to boost accuracy. Ignore
for now.)*

---

## 6. The `fluke_embeddings` table (the vector DB)

```sql
CREATE TABLE fluke_embeddings (
    image_id      bigserial PRIMARY KEY,        -- one row per image
    individual_id bigint      NOT NULL,         -- which whale
    sighting_id   bigint      NOT NULL          -- which encounter
        REFERENCES sightings,                   -- must exist in sightings
    asset_ref     text        NOT NULL,         -- the file we embedded
    source_url    text,                         -- original URL (attribution)
    embedding     vector(512) NOT NULL,         -- the vector — width must match model
    embedder_ver  text        NOT NULL,         -- which model made it
    created_at    timestamptz DEFAULT now(),
    UNIQUE (sighting_id, embedder_ver)          -- no dupes per model version
);
```

- **`UNIQUE (sighting_id, embedder_ver)`** is the quiet hero: it lets the same
  image have a v1 *and* a v2 vector, but blocks accidental duplicates within a
  version. It's what makes "rerun safely" true at the database level.
- The fast-search **HNSW index** is added *after* the bulk load (cheaper to build
  on a full table than to maintain during 114K inserts).

---

## 7. The traps we hit (the real lessons)

These cost us the most time — and are the most useful to remember.

### Trap 1: "None of my imports work" → wrong Python interpreter
PyCharm wasn't pointed at the project's `.venv`. Also, the `.venv` had been
copied broken (every file 0 bytes). Fix: delete `.venv`, `uv sync` to rebuild,
point PyCharm at `.venv/bin/python`.
**Lesson:** "imports not found" is almost always the IDE using the wrong
interpreter, not a code problem.

### Trap 2: psycopg2 vs psycopg
Use **psycopg** (v3), not the old psycopg2. SQLAlchemy works with v3 too — the
URL scheme is `postgresql+psycopg://...`.

### Trap 3: "what does -nd mean?" (Creative Commons licenses)
CC license letters: **BY**=credit, **NC**=noncommercial, **SA**=sharealike,
**ND**=**NoDerivatives**. An embedding is arguably a *derivative*, so `-nd`
images shouldn't be embedded. Turned out we had 0 of them in the downloaded set,
so it was a non-issue — but the filter stays as cheap insurance.
**Lesson:** filter `-nd` at *embed* time (it's a license rule), not download.

### Trap 4: text vs bigint for `individual_id`
A foreign-key / join column must be the **same type** on both sides. Postgres
does **not** silently compare `text` to `bigint` — it errors. We made everything
`bigint`. (The conversion *succeeding* proved all IDs were numeric.)

### Trap 5: "relation 'manifest' does not exist" (the big one)
The script connected to the **wrong database** (`wildfire`, leftover from the
seed project) instead of `oceans`. Worse, the tables were in the `datasets`
**schema**, not `public`, so the connection's `search_path` couldn't find them.
Fix: connection URL =
`postgresql://localhost:5432/oceans?options=-csearch_path%3Ddatasets,public`.
**Lesson:** "table doesn't exist" but it works in your SQL console = you're
connected to a *different database/schema* than you think.

### Trap 6: PyCharm worked, terminal didn't → a stale shadow file
A **copy of `config.py` was sitting in `.venv/.../site-packages/`** and
*shadowing* the real `src/config.py`. So edits to the real file did nothing in
the terminal. Caused by a `force-include` in `pyproject.toml` that physically
copies files during editable installs.
Fix: replaced `force-include` with `sources = ["src"]` + `include = [...]`, so
editable installs stay path-based (a pointer to `src/`, no copies). Verified it
survives `uv sync`.
**Lesson:** if `import x` behaves differently in the IDE vs terminal, check
`print(x.__file__)` — something may be shadowing your source.

### Trap 7: "No such file: blobs/.../6119.jpg" → relative paths + cwd
`manifest.asset_ref` is a **relative** path (`blobs/4606/6134.jpg`), written
relative to `src/data/`. `Image.open` resolves it against your *current working
directory*. PyCharm ran from `src/data/` (worked); the terminal ran from the
project root (failed). Fix: anchor to an absolute base derived from the script's
own location: `BLOB_ROOT = Path(__file__).resolve().parent`, then open
`BLOB_ROOT / asset_ref`.
**Lesson:** never trust relative paths in a script you might run from anywhere —
anchor them to `__file__`.

---

## 8. Where things stand

- ✅ Images downloaded to `src/data/blobs/`
- ✅ `manifest` table populated, statuses normalized to `'ok'`
- ✅ `fluke_embeddings` table created
- ✅ Model deps installed (`torch`, `open_clip_torch`), Mac GPU (MPS) working
- 🔄 **Embedding the full ~114,093 images** (running)
- ⏭️ Next: `photo_id.py` (the nearest-neighbor query), then the HNSW index, then eval

---

## 9. Cheat sheet

**Run the embed job (from anywhere):**
```bash
uv run python src/data/embed_images.py
```

**Watch progress (target = 114,093):**
```sql
SELECT count(*) FROM datasets.fluke_embeddings WHERE embedder_ver = 'clip-vitb32-v1';
```

**The connection string** (in `src/config.py`):
```
postgresql://localhost:5432/oceans?options=-csearch_path%3Ddatasets,public
```

**If the terminal ever can't find tables again** (after a reinstall recreates a shadow):
```bash
rm -f .venv/lib/python3.12/site-packages/{config,logging_config,exceptions}.py
```
(Should no longer be needed after the `pyproject.toml` packaging fix, but kept here just in case.)

**Add the fast-search index — only AFTER the full embed finishes:**
```sql
CREATE INDEX ON datasets.fluke_embeddings USING hnsw (embedding vector_cosine_ops);
```

**Key terms one-liner:** *embedding* = image→512 numbers · *CLIP* = the model ·
*cosine* = closeness measure · *embedder_ver* = which model made the vector ·
*idempotent* = safe to rerun.
