# Design note — the by-individual re-ID eval (`reid_split`)

*Status: design, not built. Skeletons below; no implementation yet.*

## 1. What this eval answers (and how it differs from what we have)

We already have a working eval: `eval.py` + `dataset.py` run a **closed-set, by-sighting
leave-one-out over ALL whales** (`build_split.sql` → `photo_id_eval_split`). It answers
*"can the embedding re-spot a whale it has photos of?"* — a memorization-friendly question,
and it's our **frozen-CLIP baseline number**.

This new eval answers the **honest generalization** question:

> *Can the embedding re-identify a whale that was **never in training**?*

It reads the frozen `datasets.reid_split` (by-**individual** disjoint split). The trainer
consumes `train`; this eval consumes **`val`** while we iterate on LoRA, and **`test`**
exactly once at the end. The number it produces is what we report as the LoRA lift.

**Two evals, side by side, deliberately.** We are not replacing the closed-set baseline —
it stays as the easy lower-bar reference. This one is the hard bar.

## 2. The one design problem that matters: gallery scoping

Everything else is plumbing. The real decision is: **what pool does a query search against?**

The disjoint protocol (see `build_reid_split.sql` header, lines 24–28) requires:

> a `test` query is matched **only against OTHER `test` whales** — its own sighting excluded.

But today's tool searches the *whole* catalog:

```python
# tools/photo_id.py — _search_nearest_images
WHERE fe.embedder_ver = %s
  {exclude_clause}          # drops only the one query sighting
ORDER BY fe.embedding <=> %s
LIMIT %s
```

So a `val` query could currently rank a `train` whale or a `test` whale as #1 — identity
leakage *at eval time*, which would silently inflate the score. **The gallery must be
restricted to the same split as the query.**

### 2a. Where the restriction lives — opinion: in the Task, not the Tool

The `photo_id` tool is a general Layer-3 perception tool. It must **not** learn about
`reid_split`, `val`, or `test` — those are eval-only concepts. Coupling the tool to the
eval split table would be an altitude violation.

**Decision:** the tool gains a *general* "search within this candidate set" capability;
the **eval** owns the split knowledge and supplies the set.

```python
# tools/photo_id.py  (SKELETON — signature change only)
def query_by_vector(
    self, qvec, k=5, filters=None, exclude_sighting_id=None,
    restrict_individual_ids: Sequence[int] | None = None,   # NEW: scope the gallery
) -> PhotoIDResult: ...
```

Threaded into SQL as a single array param (not a giant IN-list):

```sql
-- _search_nearest_images, when restrict_individual_ids is given
AND fe.individual_id = ANY(%s)      -- %s := list of allowed individual_ids
```

`= ANY(array)` keeps it one bound parameter regardless of gallery size, and it's the
honest seam: the tool just searches a candidate set; the eval decides what that set is.

> This reuses the *spirit* of the dormant `filters` param but not its mechanism —
> `Filters` means species/region/date (real-world query narrowing). Split-scoping is an
> eval artifact, so it gets its own explicit param rather than overloading `Filters`.

### 2b. ANN + filter recall — opinion: go exact for the eval

Subtle trap. HNSW returns the top-N nearest over the index, **then** the `= ANY` filter
drops the ones outside the split. `test` is ~15% of whales, so of `N_CANDIDATE_IMAGES=50`
nearest, only ~7 survive on average — too thin for a stable top-5, and the survivors aren't
even guaranteed to be the *true* nearest within the split (classic post-filter recall loss).
That would make the eval number a function of index approximation, not of the embedding.

**Decision:** for this eval, **prefer exact over fast.** The per-split gallery is small
(one split's whales, a few thousand images), and we run this a handful of times, not in a
hot loop. Let the filtered query do an exact cosine scan over the eligible subset — the
planner will likely skip HNSW, which is *correct* here. A measurement tool should not
inherit the approximation error of a latency optimization.

Concretely that means: when `restrict_individual_ids` is set, don't worry about whether
HNSW is used; just make sure the candidate pool is the full eligible set (raise the image
cap or drop the `LIMIT` before aggregation, then take top-k). Verify recall by spot-check:
the true whale's *other* photos must always be reachable in the pool.

## 3. Components and altitude

```
ReIDSplitDataset(split=...)      # NEW  — reads reid_split, emits Cases (query whales)
        │  query_sighting_id, individual_id, name
        ▼
PhotoIDTask(gallery_individual_ids=...)   # EXTEND — passes the gallery scope to the tool
        │  fetch stored qvec → query_by_vector(restrict_individual_ids=...)
        ▼
RetrievalRanking (reid@1, reid@5, mrr)    # REUSE verbatim — Span = individual_id equality
        ▼
eval.py  --split val|test                 # EXTEND — pick split, enforce the discipline
```

- **Dataset** (`ReIDSplitDataset`): pure projection of `reid_split`, exactly like today's
  `PhotoIDLeaveOneOut` projects `photo_id_eval_split`. One case per eval whale, keyed on its
  deterministic `query_sighting_id`. **Stays stochastic-free** — re-seeding is reproducible.
- **Task**: same as today (fetch the stored query vector, search with it excluded), **plus**
  it now also computes/receives the gallery set for the active split and forwards it. The
  gallery set is `{individual_id : split = <this split>}` — load it once per run, not per case.
- **Evaluators**: **no change.** `RetrievalRanking` with `Span(path=str(individual_id))`
  already gives Recall@1 / Recall@5 / MRR. This is the payoff of the closed-set design — the
  generalization eval reuses the scorer verbatim.

## 4. Opinionated smaller decisions

1. **`test` needs an explicit guard.** Make `--split test` require an extra `--final` flag
   (or refuse without it), so you cannot *accidentally* spend the test set while iterating.
   The discipline (val = react freely, test = once) should be enforced by the CLI, not by
   willpower. Default `--split val`.

2. **Embedder version is the A/B axis, and it's free.** Split membership is keyed on
   `individual_id` / `sighting_id` — **embedder-independent**. So the baseline run and the
   LoRA run score the **same** `test` whales; only the vectors differ. This already works via
   `Settings.embedder_ver` / `PhotoIDTask.label`. The dataset does **not** get re-seeded per
   embedder — only the Task's `ver` changes. Confirm both versions have embeddings for every
   eval whale before scoring.

   **The fair baseline is HF, not open_clip.** LoRA runs on **HuggingFace** CLIP (peft can
   target its `q/k/v/out_proj`; open_clip fuses them — see the tutorial). So the apples-to-
   apples pair is **`clip-hf-vitb32-v1`** (HF transformers, adapters OFF) vs
   **`clip-hf-vitb32-lora-v1`** (same HF base + adapters) — *not* the original open_clip
   `clip-vitb32-v1`. Same OpenAI ViT-B/32 weights, same 512-d, same framework → the only
   thing that moves is the adapters. (The open_clip `clip-vitb32-v1` number stays as a
   loose reference, but the *reported lift* is HF-base → HF-LoRA.) Per the checklist, this
   eval's baseline embedder_ver is `clip-hf-vitb32-v1`.

3. **Abstain is OFF for the headline metric.** Within a split, the query's true whale is
   *always* present in the gallery (we held out one photo, kept the rest) — it's closed-set
   by construction. So `abstain`/`NOVEL` is not part of Recall@k here; report ranking only.
   Threshold calibration (`ABSTAIN_THRESHOLD`, currently a 0.80 placeholder) is a **separate
   val-only analysis** — fit it on `val`, never on `test`.

4. **Headline metric stays Recall@1 / Recall@5 / MRR**, not Precision@k (one correct whale
   caps precision@5 at 0.2). Same rationale as the baseline eval's docstring.

5. **One `eval.py`, parameterized — don't fork a second script.** Add `--split` and a
   dataset selector; keep the LangSmith seeding/verify path. Two eval *datasets*, one runner.

## 5. Skeletons

```python
# evals/photo_id/reid_dataset.py  (SKELETON)
@dataclass
class ReIDSplitDataset:
    """DatasetSource over datasets.reid_split — one case per val/test query whale."""
    split: str = "val"                  # 'val' (iterate) | 'test' (final, guarded)
    name: str = "photo-id-reid-byindividual"
    version: str = "v1-clip-vitb32"     # embedder-independent split; ver tags the catalog

    def __post_init__(self):
        assert self.split in ("val", "test")
        self.name = f"{self.name}-{self.split}"

    def gallery_individual_ids(self, gw) -> list[int]:
        """The candidate pool: every individual in THIS split (the disjoint gallery)."""
        ...  # SELECT individual_id FROM datasets.reid_split WHERE split = %s

    def load(self) -> list[Case]:
        """One case per row WHERE split=self.split AND query_sighting_id IS NOT NULL."""
        ...  # JOIN individuals for name; key id on query_sighting_id; tag with split
```

```python
# evals/photo_id/task.py  (SKELETON — extension)
class ReIDTask(PhotoIDTask):           # or add a gallery param to PhotoIDTask
    def __init__(self, *, gallery_individual_ids: Sequence[int], **kw):
        super().__init__(**kw)
        self._gallery = list(gallery_individual_ids)   # scope, loaded once

    async def run(self, task_input):
        sid = task_input["query_sighting_id"]
        qvec = self._fetch_vector(sid)
        if qvec is None:
            return None, Usage()
        result = self.tool.query_by_vector(
            qvec, k=self.k, exclude_sighting_id=sid,
            restrict_individual_ids=self._gallery,      # <-- the disjoint gallery
        )
        return result, Usage(total_tokens=0)
```

```python
# evals/photo_id/eval.py  (SKELETON — wiring)
ap.add_argument("--split", choices=["val", "test"], default="val")
ap.add_argument("--final", action="store_true", help="required to run --split test")
# ... if args.split == "test" and not args.final: refuse.
# dataset = ReIDSplitDataset(split=args.split)
# task = ReIDTask(gallery_individual_ids=dataset.gallery_individual_ids(gw), k=args.k)
# (RetrievalRanking evaluators + runner: unchanged)
```

## 6. Build order (when we implement)

1. Tool: add `restrict_individual_ids` to `query_by_vector` / `_search_nearest_images`;
   confirm exact recall over a scoped pool (the ANN trap in §2b).
2. `ReIDSplitDataset` + `gallery_individual_ids`.
3. `ReIDTask` (or `PhotoIDTask` gallery param).
4. `eval.py` `--split`/`--final` wiring.
5. Run `--split val` on the **HF base** (`clip-hf-vitb32-v1`, adapters off) → record the
   generalization baseline. (LoRA reruns later flip `embedder_ver` to `clip-hf-vitb32-lora-v1`;
   `test` is touched only at the very end.)

## 7. Open questions

- Gallery scope when the agent uses the tool for real (not eval): the dormant `Filters`
  path (species/region) is the production analogue of scoping. Worth reconciling later so the
  tool has *one* candidate-restriction story, not two.
- Does `val` have enough whales for a stable metric, or should we report a confidence band?
  Check the `reid_split` sanity-query counts before trusting a single val number.
