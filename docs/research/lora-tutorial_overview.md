# The OCEANS photo-ID pipeline — overview

The story end-to-end, from raw data to the LoRA experiment. This is the map to
come back to when the pieces stop fitting together in your head.

## The pipeline (one line)

```
download → embed catalog → [photo_id tool] → eval baseline ──┐
                  ▲                                            │ measure lift
                  └──── LoRA fine-tunes the embedder ──────────┘
   (re-embed under a NEW embedder_ver, re-run the SAME eval, compare)
```

The embedder is the **hinge**: the catalog build *and* the query tool both run
images through it. LoRA changes that embedder → better vectors → you re-embed and
re-score. That is the entire project arc.

---

## The story

### 1. The raw data — OBIS-SEAMAP
We started from an OBIS-SEAMAP dataset: rows containing **image URLs** and **text
identifications** (which whale, plus license/source metadata). This is the seed.

### 2. Shape it into a domain model
We turned that flat dump into two real tables — **`individuals`** (one row per
known whale) and **`sightings`** (one row per encounter, carrying the image URL,
license, and a foreign key to its individual) — and cleaned the data along the way
(dropped junk rows, normalized identities, applied the license filter so we never
train on no-derivatives images).

### 3. Download the images — `data/download_images.py`
We read the URLs out of the `sightings`/`manifest` rows and fetched each image
from Happywhale's CDN (politely — 6 worker threads, retries, timeouts). Each file
landed on disk keyed by its identity: **`blobs/<individual_id>/<sighting_id>.jpg`**.
The folder is the *individual*, the filename is the *sighting* — so the path
itself encodes "which whale, which encounter." A `manifest.csv` recorded the
outcome (`ok` / failed) per row.

### 4. Embed the catalog — `data/embed_images.py`
We walked every successfully-downloaded image, ran it through an **open_clip**
model, and stored the resulting **512-d vector** alongside its text identity in the
**`fluke_embeddings`** table (`individual_id`, `sighting_id`, the `vector(512)`,
and an `embedder_ver` stamp).

**How it works for openCLIP:**
`models/image_embedder.py` is the single source of truth. `ImageEmbedder(ver)`
looks up `ver` in the `EMBEDDERS` registry → gets an
`EmbedderSpec("ViT-B-32", "openai", 512)` →
`open_clip.create_model_and_transforms(...)`. For each image: open → preprocess
with the model's *own* transform → `encode_image` → **L2-normalize** so cosine
distance is meaningful. A guard asserts the output is exactly 512-d before anything
gets written. `embed_images.py` is just plumbing around it (pick pending rows →
embed in batches of 32 → insert), and it is **idempotent per version**: it skips
sightings already embedded at this `embedder_ver`.

**How we add bioCLIP as an option:**
bioCLIP is *also* a native open_clip model, so it is a one-line registry entry:
`"bioclip-v1": EmbedderSpec("hf-hub:imageomics/bioclip", None, 512)`. Same 512-d, so
the column is unchanged. To switch, you do not edit code — you set
**`EMBEDDER_VER=bioclip-v1`** (env or `.env`); `Settings.embedder_ver` flows down
through the embedder and the row stamp. Re-running `embed_images.py` then embeds the
whole catalog under the new tag *alongside* the old vectors (different
`embedder_ver`), because the table's unique key is `(sighting_id, embedder_ver)`.
That coexistence is what lets us A/B.

### 5. Move the blobs out of the repo
We moved `./blobs` to **`~/Source/DATA/oceans/images`** (same
`<individual_id>/<sighting_id>.jpg` layout, just outside the code tree). The path is
now a config setting (`Settings.image_root`, override `IMAGE_ROOT`), and we fixed a
`.gitignore` rule that had been silently hiding the pipeline source. The legacy
`blobs/` prefix on `asset_ref` gets stripped at join time.

### 6. Split train vs. test — the real decision
The key question in re-identification is **what you hold out**, because it defines
what "good" even means:

| You hold out… | What the test asks | Difficulty | Realism for conservation |
|---|---|---|---|
| **Sightings** (closed-set) | "Hide one photo of a *known* whale — do its other photos still rank it #1?" | Easier | Re-spotting a catalogued whale |
| **Individuals** (open-set) | "Here's a whale the model *never saw* — can it tell it's NOVEL?" | Harder | A brand-new whale shows up |

What we have **already** frozen (`evals/photo_id/build_split.sql`) is the
**closed-set, leave-one-out by sighting** version:

- Eligible = individuals with **≥10** photos (so ≥9 remain as a "gallery" after we
  hide one).
- For each, pick **one** query sighting **deterministically**
  (`row_number() ORDER BY md5(sighting_id)`) — reproducible forever on the same
  catalog, and unbiased.
- The query sighting stays *in* the table; "held out" means we pass
  `exclude_sighting_id` at query time so it cannot trivially match itself.
- It is stored in `datasets.photo_id_eval_split` as the **fixed yardstick**, and the
  `role` column already reserves a seam for an `open_set_novel` slice later.

The load-bearing rule for **LoRA (Phase-C)**: those held-out query sightings **must
be excluded from training**, or the measured lift is fake (the model would have
memorized the test). The split table is exactly that guardrail.

> Note: `build_split.sql` currently hardcodes `embedder_ver = 'clip-vitb32-v1'` to
> enumerate eligible individuals. That is fine — the split is a set of *sighting
> IDs*, which are model-independent — but worth knowing it is pinned to the baseline
> catalog.

---

## File map

| Stage | File | Class / key fn | What it does | Role in the LoRA arc |
|---|---|---|---|---|
| **Shared core** | `config.py` | `Settings`, `apply_hf()` | Central config: `embedder_ver`, `image_root`, `hf_token` | `embedder_ver` is the A/B switch between baseline and LoRA vectors |
| | `models/image_embedder.py` | `ImageEmbedder`, `EMBEDDERS`, `resolve_spec()` | Image → 512-d unit vector; one source of truth | **The thing LoRA fine-tunes.** Today loads frozen CLIP; later loads CLIP+adapters |
| | `stores/postgres/gateway.py` | `get_pg_gateway()` | DB access to `fluke_embeddings` | Where both vector versions live side by side |
| **1. Build catalog** (offline) | `data/download_images.py` | `Task`, threaded fetch | Pull fluke JPEGs + write `manifest.csv` | Produces the raw images, once |
| | `data/embed_images.py` | `main`, `read_pending`, `write_batch` | Embed every image → `fluke_embeddings`, stamped `embedder_ver` | **Re-run per version** — how a LoRA model gets a full catalog |
| **2. Serve** (roadmap item 5) | `tools/contracts.py` | `ToolResult`, `Filters` | The "dumb tool" interface | Tool stays identical across versions |
| | `tools/photo_id.py` | `PhotoIDTool`, `_search_nearest_images` | Query image/vector → ranked individuals (cosine kNN + abstain) | The system-under-test; reads vectors of the active `embedder_ver` |
| **3. Measure** (roadmap item 6, "Layer-A") | `evals/photo_id/build_split.sql` | — | Freeze the held-out leave-one-out split (once) | Fixed yardstick so baseline vs LoRA are comparable |
| | `evals/photo_id/dataset.py` | `PhotoIDLeaveOneOut`, `load()` | Project the frozen split → `Case`s | Same cases every run |
| | `evals/photo_id/task.py` | `PhotoIDTask` | Run the tool on one held-out sighting (vector-only) | `label = photo_id-<ver>` → each version gets its own experiment |
| | `evals/photo_id/eval.py` | `main` | Score Recall@1 / Recall@5 / MRR over all cases | **Produces the number you compare** |
| | `evals/photo_id/trace_one.py` | `main` | Teaching trace of one case, stage by stage | Shows *why* raw CLIP scores low (the gap LoRA closes) |
| | `evals/framework/*` | `RetrievalRanking` | Reused metric harness | Unchanged; just scores ranked lists |
| **4. LoRA** | `training/lora_sanity.py` | `wrap_with_lora`, `main` | **Part 0**: prove adapters attach, base frozen, grads flow | The wiring check — precondition for real training |
| | *(future)* `training/lora_train.py` | — | Phase-C: actually fine-tune the adapters on flukes | Does not exist yet — the next real build |

---

## Where you are, and what is next

- **Done:** data model → download → embed catalog → retrieval tool
  (`tools/photo_id.py`) → LoRA *wiring* sanity (`training/lora_sanity.py`, Part 0).
- **Next up (item 6):** run `evals/photo_id/eval.py` to get the **baseline number**
  (Recall@1/5, MRR) on `clip-vitb32-v1`. We *expect* it to be mediocre — raw CLIP
  barely separates individual whales (you saw this in `trace_one.py`: scores cluster
  ~0.95 across *different* whales).
- **Then (Phase-C):** train LoRA adapters → register as a new version (e.g.
  `clip-vitb32-lora-v1`) → re-embed under that `EMBEDDER_VER` → re-run the *same*
  eval → the gap between the two Recall@1 numbers **is** the LoRA lift.

The refactors done this session (embedder_ver through Settings, the dim guard,
`image_root` config, `apply_hf`) exist to make that final comparison clean and
code-free: you hold baseline and LoRA vectors in one table and flip between them
with an env var.
