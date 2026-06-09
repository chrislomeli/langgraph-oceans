# Ocean-Mammal Conservation Agent — Vision Note

> **STATUS: BUILD UNDERWAY.** The data + photo-ID retrieval layer is being built
> now; the agent spine and eval follow. This note pins the decisions firmed up so
> far and the questions still open. Hold it loosely; revise freely.

---

## Build roadmap (technical work items)

> **How to use this:** the **next thing to build is always the first row not
> marked ✅.** No need to ask "what's next" — read down the Status column.
> Legend: ✅ done · 🔄 in progress · ⏭️ next · ⬜ later.

| # | Feature (what we build) | Technology (how) | Problem it solves (why) | Status |
|---|---|---|---|---|
| | **Phase A — Photo-ID retrieval** *(the separable exhibit)* | | | |
| 1 | Relational + geo data load | Postgres + PostGIS (`datasets` schema) | Ground-truth identities, sightings, locations, vessel traffic for the agent to reason over | ✅ |
| 2 | Image acquisition | Python threaded downloader → `blobs/` + `manifest` table | Gets fluke photos onto disk, each linked to its individual/sighting | ✅ |
| 3 | Image embedding | CLIP ViT-B/32 (PyTorch/MPS) → `pgvector(512)`, stamped `embedder_ver` | Turns each photo into a comparable 512-d "fingerprint" vector | ✅ |
| 4 | Vector index | pgvector **HNSW** (cosine / `vector_cosine_ops`) | Makes nearest-neighbor search fast (ms) instead of scanning all 114K rows | ✅ |
| 5 | `photo_id` retrieval | Cosine kNN query, aggregate-by-individual, calibrated **abstain** threshold | "Which catalogued individual is this photo?" → ranked candidates + confidence + `NOVEL` | ✅ |
| 6 | Eval — **Layer A** (retrieval) | Held-out split; Precision@k / MRR; open-set abstain test | Proves the retriever is accurate *and* won't hallucinate an identity | 🔄 |
| | **Phase B — The agent spine** *(where "agentic" is earned)* | | | |
| 7 | Text corpus ingestion | Chunk + text embedder → pgvector (separate **text** vector space) | Builds the knowledge to answer "what's *known* about this individual" | ⬜ |
| 8 | Tool contracts | Pydantic `BBox` / `Filters` / `Citation` / `ToolResult` | One typed, "dumb" interface the agent learns once; tools never decide control flow | ⬜ |
| 9 | Retrieval tools | `hybrid_search`, `sighting_lookup`, `vessel_traffic`, `catalog_search` | The agent's capabilities: text RAG, sighting history, AIS range-overlap | ⬜ |
| 10 | Agent graph **v1→v2→v3** | LangGraph router + ReAct loop + recovery edge | The judgment layer: route · pick tool · judge sufficiency · disambiguate · recover — **the agency** | ⬜ |
| 11 | Grounded synthesis | LLM synth role + `Citation` provenance | Produces the cited answer with grounding that's *measured*, not assumed | ⬜ |
| 12 | Eval — **Layer B + C** | Faithfulness scoring + orchestration-trace scoring | Proves answers are grounded **and** the agent routed/recovered correctly ("agentic = a number") | ⬜ |
| | **Phase C — Optional ML lift** *(the personal learning goal)* | | | |
| 13 | LoRA fine-tune embedder | Metric learning (ArcFace/triplet) + **LoRA** → re-embed as `...-v2` | Specializes CLIP for fine-grained fluke ID → accuracy lift over baseline, measured against Layer A | ⬜ |

**Reading the phases:** Phase A is a standalone, benchmarkable photo-ID exhibit
(finishable on its own). Phase B is the orchestration+eval spine that turns it
into a research *agent* (the center of gravity). Phase C is bounded, optional ML
that improves one tool's internals without becoming "an ML project." Build A → B,
slot C in whenever — it re-embeds into a new `embedder_ver`, so it never blocks
the rest.

---

## Elevator pitch (agency-forward)

> A research agent for marine mammals that **decides how to answer**, not just
> what to retrieve. Ask it about a whale in a photo and it identifies the
> individual, *reasons about the uncertainty* — disambiguating look-alikes,
> recovering when the animal isn't in the catalog — chains across photo,
> sighting, and traffic data to follow the question wherever it leads, and
> answers with citations whose grounding is **measured**, not assumed. The
> storage and retrieval are plumbing; the agent is the judgment on top — and the
> eval harness proves it.

*Honest-framing line (keep attached):* Built as a portfolio demonstration —
real public data and established methods, applied end-to-end with
production-style evaluation; not a novel research result or a deployed product.

---

## Why this project

A portfolio piece for an **agentic engineer who uses RAG/ML fluently** — the
same identity the code-intel project serves, retargeted at a real domain.

The portfolio bar (settled): at this level, "solves a real-world problem" means
**real public data + an honest, reproducible result using established methods** —
*not* novelty. Claiming novelty would hurt credibility. The win is "I applied the
right tools competently to a real, meaningful domain, framed with humility."

This can be posited as a **portfolio, not a product** — exhibits don't have to
fuse into one cohesive product, *provided* they share connective tissue:
- **One domain** (ocean mammals), and
- **One architectural spine** (the agent + the two-layer eval framework).
That guardrail is what keeps it from reading as "a pile of tutorials."

Follows the standing principle: build framework-grade, reuse across the next
projects; exclude only on discipline/misuse, never on project size.

---

## The three anchors — where they landed

### Anchor 1 — Multimodal modality → **VISUAL (photo-ID)** ✅
Identify individual whales/dolphins by fluke/dorsal-fin patterns
(re-identification). Chosen over audio because it's **lower-tension** and a
better fit for the existing stack:
- Re-ID *is* embedding + nearest-neighbor = **reuses the vector-search spine**.
- Scoring **reuses the Layer-A retrieval scorer verbatim** — Precision@k / MRR
  over ranked NN matches (image anchors instead of `file:symbol`).
- Clean, benchmarkable data (Kaggle Happywhale); trivial preprocessing vs audio's
  DSP weeds; CLIP-style text↔image multimodal is native to vision.

*Audio (passive acoustic monitoring) — considered, not chosen.* Higher tension
(DSP preprocessing, messy/imbalanced labels, detection-not-retrieval framing,
younger text↔audio models). **But noted for the record:** audio is
*irreplaceable* for cryptic/deep-diving species (beaked whales), 24/7 unattended
monitoring, and real-time ship-strike buoys. If a future problem needs hearing
the unseeable, audio is the right (sometimes only) tool.

### Anchor 3 — Agent's job → **knowledge/orchestration agent as the SPINE, photo-ID as a subordinate tool** ✅
Not either/or. Top priority is **agentic orchestration + eval rigor**; light ML
is welcome but *never at its expense*. So:
- The agent's job is **reason → retrieve → synthesize → cite**, scored by the
  two-layer eval. That spine is the center of gravity.
- **Photo-ID is a callable tool inside the toolbelt** — structurally subordinate,
  so it cannot eat the spine.

### Anchor 2 — ColPali / visual-document RAG → **DEFERRED (stretch flex)** ⏸
Multimodal credential already comes from photo-ID, so ColPali would be a *second*
multimodal technique whose cost is **multi-vector late-interaction storage** —
infra pgvector doesn't do natively, exactly the kind of thing that would compete
with the agentic+eval priority.
- **Now:** don't build.
- **Cheap fallback if the corpus is figure-heavy:** the **caption-bridge** —
  VLM-captions each figure/map/chart, embed the text (reuses the text stack,
  near-zero infra).
- **Later:** ColPali becomes a clearly-labeled retrieval-depth exhibit added
  *after* the spine is solid, if wanted.

---

## Is this actually agentic? — the litmus test

A skeptic's challenge to take seriously: *"This is mostly storage and retrieval.
Why is it even an agent? What use is AI here?"* The skeptic is **right about the
trivial version** — a single-shot `photo → look up that ID → summarize` flow is
a hardcoded pipeline (the v1 linear graph), and calling it an agent is marketing.
Plain RAG (`embed → retrieve → generate`) is a DAG, not a decision-maker. The
agentic claim lives only at **v2 (decomposition) and v3 (ReAct + fallback)** — so
this project must reach v3 to earn the word.

**What makes it agentic:** the model owns the control flow, and *you couldn't
have written that flow in advance.* The next tool call is computed from the last
result; the agent iterates on uncertainty, falls back adaptively, decomposes and
routes. The intelligence isn't in the retrievals — it's in the **decisions
between** them. Two AI layers carry it: **perception** (the learned photo-ID
embedding — can't be rule-written; the LoRA lives here) and **reasoning** (the
LLM deciding which tool, whether evidence suffices, how to disambiguate, when to
recover). Crucially, the **eval harness measures** the orchestration — correct
routing, recovery from a no-match, grounded synthesis — so "it's agentic" is a
number, not a vibe.

**Photo-ID's built-in uncertainty is the fuel.** It hands you questions that
*can't* be a straight-line script:
- **Disambiguation loop** — 3 candidates at 0.6 confidence → the agent pulls each
  candidate's last-known location, cross-references the report's date/place,
  eliminates or asks for a second photo. The disambiguating query is built from
  the photo-ID result.
- **Multi-hop (step N needs step N-1)** — *"is this individual's range over
  high-traffic shipping lanes?"* → ID → retrieve its recent sightings → extract
  *that* range → query vessel traffic *for that range* → synthesize. The AIS
  query is unwritable in advance.
- **Open-set fallback** — no confident match → likely uncatalogued → switch
  strategy, search text reports for recent unmatched-individual descriptions in
  that area, report "probably new, here's corroboration." Pure recovery.
- **Conflict adjudication** — two reports disagree on abundance → notice it,
  retrieve more or flag it, rather than blindly averaging.

### The litmus test (apply to every demo question)

> *"Could I have written the exact sequence of tool calls in advance?"*
> - **Yes** → it's a pipeline; the skeptic wins; don't showcase it as agentic.
> - **No — the path depends on what came back** → that's an agentic exhibit.

**Design commitment (made now, not bolted on later):** build the demo set
*entirely* from "No" questions. That single discipline is what makes the pitch
survive the skeptic.

---

## LoRA fine-tuning (a personal learning goal) — bounded home

Fine-tune the **photo-ID embedder** (metric learning on Happywhale) with LoRA.
Self-contained, benchmarkable against a leaderboard, and it's *one tool's
internals* — so the "fun, rewarding, light ML" stays garnish and cannot
metastasize into "this became an ML project." Everything else stays pretrained
/ off-the-shelf.

---

## The coherent flow (proves it's not contrived)

> *"Here's a photo from a sighting report — which catalogued individual is this,
> and what's known about it?"*
> agent → **photo-ID tool** (image → individual ID; LoRA embedder + vector NN) →
> **RAG** over that individual's sighting history / stock-assessment context →
> **grounded, cited answer**, faithfulness-scored.

Image-in → identity → knowledge retrieval → synthesized cited answer: a legible,
multi-tool agentic trace.

**Corpus design requirement:** the flow only works if the corpus has
**individual-level** content (sighting histories, named individuals). Design the
corpus around that — it's a design choice, not a hard problem.

> ✅ **Confirmed available (recon 2026-06-04):** individual-level content is real
> and open after all. OBIS-SEAMAP hosts **Happywhale-contributed datasets** with
> `organism_id` (the individual), `external_resource` (the fluke photo), and real
> lat/lon/date — one namespace, CC0-dominated license, ~32k humpback individuals
> (20k+ with multi-sighting histories). So the flow stays **per-individual** as
> originally designed: photo-ID → that individual's sighting history → range →
> AIS. No synthetic data, no species-level fallback needed. See
> `docs/research/data-research.md` (CORRECTED) and `data-build-plan.md`.

---

## De-risking: build the tool separable

Build the photo-ID tool as a **standalone, separable exhibit** that *also* wires
into the agent. Two outcomes for one effort:
- **Integration smooth →** unified "research agent that IDs individuals and
  answers grounded questions" (the impressive version).
- **Runway runs out →** two loosely-coupled exhibits (the orchestration+eval
  agent *and* a benchmarked photo-ID+LoRA notebook) sharing domain + eval
  framework. "Portfolio not product" sanctions this; fully finishable.

Lead with the spine; let integration be the upside, not the prerequisite.

---

## High-level architecture (layers, top-down)

```
1. Interface        CLI: question (+ optional image[s]) → answer + citations + decision trace
2. Orchestration    LangGraph router-fronted ReAct agent.  ★ ALL agency lives here
                    (route · pick tool · judge sufficiency · disambiguate · recover · synth)
3. Tools            photo_id · hybrid_search · sighting_lookup · vessel_traffic · catalog_search
                    (typed, deliberately "dumb", one shared filter contract)
4. Retrieval/Index  TEXT vector space · IMAGE vector space · relational (individuals,
                    sightings) · blob store (images)
5. Models           LLM roles (synth · judge · router) · text embedder ·
                    image embedder (LoRA) · prompt registry
6. Ingestion        offline index build; pinned snapshot for eval stability

cross-cutting:  Evaluation (A: retrieval · B: answer · C: orchestration) + LangSmith
external:       Happywhale/Flukebook · NOAA · OBIS-SEAMAP · AIS
```

**Three load-bearing principles**
- **Agency lives in exactly one layer (2); tools (3) are dumb.** `photo_id`
  returns ranked candidates + confidence + an abstain outcome; it does NOT decide
  what to do about low confidence. That decision is the agent's. This is the
  whole answer to the "why is it agentic" skeptic, and it keeps tools
  unit-testable.
- **Two vector spaces (text + image), never mixed.** Both in pgvector, queried
  via different tools, fused in the agent's *reasoning* — not in the index. Lets
  the same Layer-A scorer grade both.
- **Eval grows a third concern, Layer C — orchestration.** Did the agent route
  correctly, take the right number of hops, recover from a no-match, avoid
  calling tools it didn't need? This is what defends "agentic" with a number;
  it's the genuinely new eval design work.

### Resolved design decisions
- **AIS / `vessel_traffic` is in the core** (it's the cleanest multi-hop
  exhibit: range → traffic), not a stretch.
- **Closed-set first, open-set-ready contracts.** Closed-set is a *runtime mode,
  not an architecture.* The seams that make the shift free: (1) `photo_id`
  always returns calibrated scores + an explicit `abstain`/`no_confident_match`
  outcome, even when unused; (2) the agent has a recovery *edge* from day one
  (stubbed early); (3) going open-set is a data hold-out split, not code.
- **Single router-fronted graph** (the v3 shape), not separate graphs per
  question class. The router classifies the question and sets the iteration
  budget + tool affordances.

---

## Layer 2 — the agent graph (one level down)

```
                    ┌──────────── iteration cap ────────────┐
                    ↓                                        │
START → router → agent_policy ──tool_call──→ tool_executor ──┘
                    │  (LLM decides: call a tool, or done)
                    │
                    ├── (history grows) ──→ compress ──┘
                    │
                    └──done / cap hit──→ synthesize → END
```

**Nodes**
- **router** (cheap classifier role) — single-fact / multi-part / investigation
  + "is there an image?" → sets the **iteration budget** and tools in play. Does
  NOT fork graphs; it parameterizes the one graph (single-fact ≈ budget 1 = the
  v1 fast path; investigation = full loop).
- **agent_policy** (`conservation_synth`, tools bound) — the ReAct brain. Each
  turn: see scratchpad + last tool result → emit another tool call *or* "done."
  **This node is the agency**; every litmus-test "couldn't-pre-script-it"
  decision happens here.
- **tool_executor** — runs the chosen tool, writes its typed result to state.
  Dumb dispatcher.
- **compress** — when loop history passes a threshold, fold prior tool outputs
  into a running digest (the *compress* context-move).
- **synthesize** (forced-exit) — on "done" OR cap-hit, produce grounded answer +
  citations from the scratchpad. Cap-hit still synthesizes; never dead-ends.

**State (`ConservationState`)**

| Field | Written by | Purpose |
|---|---|---|
| `query`, `images` | input | question + optional photo(s) |
| `route`, `budget` | router | question class + iteration cap |
| `candidates` | photo_id | ranked individuals + scores + **abstain flag** |
| `resolved_individual` | agent | an ID, or `NOVEL`, or `None` |
| `findings` (scratchpad) | agent | accumulated evidence across hops *(write move)* |
| `messages` | loop | ReAct transcript |
| `iterations` | loop | against the cap |
| `decisions` / trace | every node | route, tool order, hops, fallback-fired |
| `answer`, `citations` | synthesize | the product |

> `decisions`/trace does double duty — it's what the CLI renders ("how I
> reasoned") AND what **Layer C** scores. Make it a first-class structured
> field, not log scraping.

**The four agentic scenarios on this one graph**
- *Disambiguation* — photo_id → 3 candidates ~0.6 → agent calls sighting_lookup
  per candidate + cross-refs report date/place → narrows `resolved_individual`.
- *Multi-hop range→AIS* — sighting_lookup → agent extracts range into `findings`
  → calls vessel_traffic(range) → synth.
- *Open-set fallback* — photo_id `abstain` → recovery edge → catalog_search over
  text → `resolved_individual = NOVEL`. (Stubbed in closed-set; edge present.)
- *Conflict* — synth sees contradictory abundance in `findings` → loops back for
  more, or flags it.

### RESOLVED — emergent vs. structured control flow

- **Disambiguation = emergent.** The agent freely decides in the ReAct loop that
  candidates are too close and it needs to disambiguate. This is the showcase of
  genuine agency; Layer C scores the *outcome* (did it resolve correctly in a
  sane number of hops), not a fixed path.
- **Open-set recovery = a structured edge.** An explicit recovery branch the
  agent routes into on `abstain`. Reliability beats flash when the claim is "it
  does NOT hallucinate a match." This is also the edge we stub first under
  closed-set, so the structure matches the closed→open staging.

---

## Layer 3 — tool contracts (one level down)

The seam between the agent and everything below it. Job: **typed in, typed out,
and the tool NEVER makes a control-flow decision** — it surfaces signals
(uncertainty, ambiguity, derived geometry) and lets layer 2 decide.

### Shared contracts (one contract the agent learns)

```python
class BBox(BaseModel):                 # geo lingua franca
    lat_min: float; lat_max: float; lon_min: float; lon_max: float

class Filters(BaseModel):              # SAME shape every tool accepts; tools ignore irrelevant keys
    species: list[str] | None = None
    region:  BBox | None = None
    date_range: tuple[date, date] | None = None
    source:  list[str] | None = None   # ["NOAA_stock_assessment","OBIS","sighting_reports",...]

class Citation(BaseModel):
    kind: Literal["document","sighting","catalog_image","ais"]
    source: str; locator: str | None = None; ref: str | None = None

class ToolResult(BaseModel):           # base every tool return extends
    tool: str
    ok: bool                           # False on empty/failure — NEVER raise into the loop
    summary: str                       # one line the agent + trace can read
    citations: list[Citation] = []
```

### Dumb-tool invariants (non-negotiable)
1. **Always return a `ToolResult`, even on empty/failure** (`ok=False`); tools
   never raise into the loop — the agent must *observe* failure to recover.
2. **Surface uncertainty; don't act on it** (`abstain`, `margin`, `density` are
   signals; the decision is layer 2's).
3. **No chaining inside a tool** — composition is the agent's job.
4. **Carry provenance** (every item → a `Citation`).
5. **Pydantic + serializable** (the LangSmith round-trip lesson).

### The five tools

```python
# 1. photo_id — the open-set-ready perception tool
def photo_id(image: ImageRef, k: int = 5, filters: Filters | None = None) -> PhotoIDResult
class Candidate(BaseModel):
    individual_id: str; name: str | None
    score: float                 # CALIBRATED similarity 0–1
    catalog: str; thumb_ref: str | None     # matched catalog image = multi-modal citation
class PhotoIDResult(ToolResult):
    candidates: list[Candidate]  # ranked desc; may be empty
    abstain: bool                # top score < threshold      ← closed→open seam
    margin: float | None         # score(top1) − score(top2)  ← emergent-disambiguation signal
    threshold: float             # echoed for eval reproducibility

# 2. hybrid_search — text RAG (reused: RetrievedChunk → TextChunk)
def hybrid_search(query: str, k: int = 8, filters: Filters | None = None) -> SearchResult
class TextChunk(BaseModel):
    source: str; title: str | None; text: str; score: float; locator: str
class SearchResult(ToolResult):
    chunks: list[TextChunk]

# 3. sighting_lookup — relational; range_bbox is the multi-hop seam
def sighting_lookup(individual_id: str, filters: Filters | None = None) -> SightingHistory
class SightingRecord(BaseModel):
    individual_id: str; date: date; location: LatLon
    region: str | None; source: str; notes: str | None
class SightingHistory(ToolResult):
    individual_id: str
    records: list[SightingRecord]    # chronological
    range_bbox: BBox | None          # COMPUTED range → feeds vessel_traffic
    last_seen: date | None

# 4. vessel_traffic — AIS (raw metrics only; risk is the AGENT's judgment)
def vessel_traffic(region: BBox, window: tuple[date,date] | None = None) -> VesselTrafficResult
class VesselTrafficResult(ToolResult):
    region: BBox; window: tuple[date,date]
    density: float; lane_overlap: bool; vessel_count: int

# 5. catalog_search — open-set recovery = scoped hybrid_search (same SearchResult)
#    catalog_search(query, k, filters=Filters(source=["sighting_reports"]))
```

The two affordances that make the resolved scenarios real: `photo_id.abstain`
(the open-set recovery edge fires on it; ignored under closed-set) and
`photo_id.margin` (small margin = close candidates = the agent's emergent
disambiguation trigger). And `sighting_lookup.range_bbox` is *computed from*
step N-1 and handed to `vessel_traffic` at step N — the unscriptable multi-hop.

### RESOLVED — Layer 3 forks
- **`ToolResult` base + subclasses** (not independent per-tool types) — uniform
  `ok`/`summary`/`citations` for the agent to reason over and Layer C to score.
- **`vessel_traffic` returns raw metrics only, no `risk_level`** — risk is the
  agent's judgment; a pre-judging tool violates the dumb-tool rule and steals the
  reasoning the pitch sells.
- **`catalog_search` is a scoped `hybrid_search`** (fixed `source` + feature
  query), same `SearchResult` — keeps the toolset lean.
- **`photo_id` calibration: validated threshold on a held-out set NOW, full
  calibration LATER.** `abstain` is meaningless on uncalibrated scores, so the
  threshold must be validated before open-set; full probability calibration can
  wait. ⚠️ Real work, not a config line — raw cosine isn't a probability.

---

## Layer 4 — the two-index schema (one level down)

The substrate the skeptic called "just storage." Design goal: keep the two
vector spaces strictly separate, but give the agent a **relational hub** to
traverse between them.

### Key insight: the relational core is the hub
The two vector spaces never touch; `individual_id` connects them.
```
  IMAGE space ──(photo_id → individual_id)──┐
                                            ↓
                                    ┌─ individuals ─┐   ← the hub
                                    └───────────────┘
                                       ↑         ↑
                          sightings ───┘         └─── documents reference it
                          (→ range_bbox → AIS)        (text space)
```
Every multi-tool trace traverses: **image vector → `individual_id` →
sightings → geo range → AIS**, with the text space queried in parallel. Vectors
stay unmixed; the FK is the bridge.

### Five stores (one Postgres, `conservation.*` schema, mirrors `code_intel.*`)
```sql
-- 1. TEXT vector space — hybrid_search + catalog_search
conservation.doc_chunks(
  chunk_id pk, source, title, locator, text,
  species text[], region geometry(Polygon,4326), obs_date date,
  individual_id text,            -- nullable: doc may reference an individual
  embedding vector(768), tsv tsvector, embedder_ver text)
  -- hnsw(embedding cosine) + gin(tsv) + gist(region)

-- 2. IMAGE vector space — photo_id (separate dim, separate index)
conservation.fluke_embeddings(
  image_id pk, individual_id fk, catalog,
  asset_ref text,                -- blob pointer, NOT the bytes
  embedding vector(512), embedder_ver text)   -- hnsw(embedding cosine)

-- 3. relational HUB
conservation.individuals(
  individual_id pk, name, species, catalog, sex,
  first_seen, last_seen, status)

-- 4. sightings — sighting_lookup; range computed from here
conservation.sightings(
  sighting_id pk, individual_id fk NULLABLE,    -- null = unmatched
  obs_date date,
  location geography(Point,4326),               -- geography = repo convention (true m distances)
  region, source, notes)                        -- gist(location)

-- 5. AIS — vessel_traffic (precomputed density grid, not raw points)
conservation.ais_density(
  cell geography(Polygon,4326), time_bucket date,
  vessel_count int, density float, lane_overlap bool)   -- gist(cell)
```
Plus a **blob store** (filesystem dir locally, object-store-swappable) for image
bytes; DB stores only `asset_ref`. Binaries stay out of Postgres.

### `Filters` → SQL (one WHERE-builder, reused across tables)
| `Filters` field | predicate |
|---|---|
| `species` | `species && ARRAY[...]` / join via individuals |
| `region: BBox` | `ST_Within(location, ST_MakeEnvelope(...,4326))` |
| `date_range` | `obs_date BETWEEN ... AND ...` |
| `source` | `source = ANY(...)` |

Same `Filters` → same builder → applied to `doc_chunks` and `sightings`. The
image table inherits species/region/date by **joining through individuals /
sightings**, not by carrying them.

### Layer-3 seams realized here
- **`range_bbox`** = `ST_Extent(location)` over an individual's sightings →
  envelope → `BBox`; `vessel_traffic` does `ST_Intersects(cell, range_bbox)`.
- **`photo_id`** = cosine NN on `fluke_embeddings`, `score = 1 - (embedding <=>
  qvec)`, `abstain = top_score < threshold`, `margin = top1 - top2`.

### Embedding versioning (migration hook)
Each vector table carries `embedder_ver`. Re-embedding writes rows with a new
version and cut-over is a filter — **migration is tracked, not a wipe.** (The
"embedding migration & versioning" scope item, landed.)

### RESOLVED — Layer 4 forks
- **PostGIS for geo** (not plain lat/lon BETWEEN). Geo *is* the domain; it's the
  standard tool, needed for `range_bbox` envelopes + AIS `lane_overlap`.
  **Already enabled in this project's database** (`ddl.sql` uses
  `geography(Point,4326)`) — free reuse alongside pgvector, not a new dependency.
  Match the repo convention: **`geography`** (spheroidal, true metre distances —
  right for ocean-scale ranges), not `geometry`. Envelope/bbox ops
  (`ST_MakeEnvelope`, `ST_Extent`) are geometry-native, so cast at those call
  sites (`::geometry`) — a minor, well-trodden wrinkle.
- **AIS = precomputed density grid**, not raw points (raw is massive; the tool
  wants aggregates; raw points are data-eng weeds for no learning gain).
- **Catalog images are the pinned corpus; query images are transient**
  (embedded live, optionally cached) — keeps the corpus stable for eval.
- **One Postgres `conservation.*` schema** (reuses the gateway, mirrors
  `code_intel.*`, repo-portable).
- **HNSW indexes** (better recall/latency than IVFFlat; pgvector supports it).

---

## Layer C — orchestration eval (the cross-cutting payoff layer)

What lets the pitch say "grounding is *measured*." Scores the **trajectory** —
not what was retrieved (A) or whether the answer is right (B), but whether the
agent *navigated correctly*. Reuses the framework wholesale: `Evaluator.evaluate
(ex) -> list[Score]`, gated-plus-companions, REPEATS sampling, injected judge.

### Metrics
| Metric | Question | Gate? |
|---|---|---|
| route accuracy | router classified right? | companion |
| **tool recall** | called the tools it *had* to? | **GATE** |
| tool precision | avoided tools it didn't need (no flailing)? | companion |
| hop efficiency | hops vs authored budget | companion (band) |
| **dependency satisfaction** | step N used step N-1's output? | **GATE** (multi-hop cases) |
| **recovery correctness** | novel individual → `abstain` fired and concluded `NOVEL`, not a fabricated match? | **GATE** (recovery cases) |
| termination | finished cleanly / forced-exit on cap? | companion |

This is **Layer A's Precision/Recall retargeted from chunks to tool calls** —
tool recall gates exactly as retrieval recall does.

### Authoring: `TrajectorySpec` (constraints, NOT scripts)
```python
class TrajectorySpec(BaseModel):
    route: Route | None = None
    required_tools: list[str] = []     # MUST call → tool recall (gate)
    forbidden_tools: list[str] = []    # MUST NOT call → tool precision
    max_hops: int | None = None        # band, not equality
    dependency: list[DepCheck] = []    # vessel_traffic.region <- sighting_lookup.range_bbox
    expect_abstain: bool = False       # novel → recovery must fire
    expect_outcome: str | None = None  # resolved_individual: an id | "NOVEL"
```
Checkable only because `decisions`/trace is a first-class structured field
(Layer 2) — this is where that decision pays off:
```python
class Step(BaseModel):
    node: str; tool: str | None; args: dict; signals: dict   # abstain, margin, ok
class Trace(BaseModel):
    route: Route; steps: list[Step]; hops: int
    terminated: Literal["done","cap","forced_exit"]; resolved_individual: str | None
```

### Evaluators
- **`TrajectoryScore`** — deterministic, the core + gate (gate = tool_recall AND
  dependency_ok AND recovery_ok). No LLM; facts about the trace.
- **`ProcessJudge`** — injected `conservation_judge` rates *decision soundness*
  ("given what each tool returned, was the next move justified?"). Depth, never
  a gate.

### Three honesty points
1. **The agent is an LLM — trajectories are NOT deterministic.** Run with REPEATS;
   report gates as **rates** ("called vessel_traffic 3/3"), and make
   **trajectory-consistency its own first-class signal** (high path variance =
   an unreliable agent = a finding). Sharp contrast to Layer A's determinism.
2. **Constraints, never exact sequences.** Scripting the path punishes valid
   alternative reasoning and *fights the agency it's measuring*. Hop budget is a
   band (`≤ max_hops`), not equality.
3. **Recovery correctness is the crown jewel.** "On N novel-individual cases the
   agent abstained and did not fabricate a match X% of the time" = *"I can prove
   it doesn't hallucinate identity."* The strongest sentence in the portfolio.

### Why three layers compose (the diagnostic)
| A | B | C | reading |
|---|---|---|---|
| ✓ | ✓ | ✓ | working |
| ✓ | ✗ faith | ✓ | navigated right, **synthesis hallucinated** → prompt problem |
| ✗ recall | ✓ | ✗ | answer right but **wrong tools** → got lucky from priors (dangerous) |
| — | — | ✗ recov | **fabricated an identity** → the failure C exists to catch |

Layer C separates "the agent's *process* is sound" from "it happened to produce
a good answer." Without it, "agentic" is a vibe; with it, a row of numbers.

### RESOLVED — Layer C forks
- **Both evaluators, decoupled by cadence.** `TrajectoryScore` (deterministic,
  gate) runs every time / in CI — no tokens. `ProcessJudge` (LLM) is added to
  the evaluator list only for periodic deep-dive runs. Same protocol, different
  cadence; never block each other (mirrors the Layer-B hand-rolled-vs-Ragas
  split).
- **Separate tool precision/recall** (recall gates; precision = efficiency
  companion), not a single F1.
- **Hop budget = band** (`≤ max_hops` passes; report actual as companion).
- **Nondeterminism → REPEATS-rates + trajectory-consistency as its own number.**
- **Disambiguation scored by outcome only** (`resolved_individual` correct within
  the hop band); don't assert it "noticed" the ambiguity (the emergent principle).

---

## Repo strategy (decided)

**Separate new repo for the oceanography conservation project, seeded by
*copying* the framework — not a shared monorepo, not a branch.**

Copy `evals/framework`, `llm`, `prompts`, `stores/postgres` (prune the
wildfire/code_intel DDL you don't need) into a fresh repo, then build oceans
there. This is the code-intel doc's "extraction is a copy + small manifest"
move, executed *now* rather than later.

**Why (the deciding concern):** the framework *will* get tweaked for oceans, and
the goal is to do that **without ever going back to keep wildfire working**, with
**drift between the two explicitly acceptable** (these are learning projects, not
products). A copy decouples them; a *separate repo* makes it absolute — breaking
wildfire from the oceans codebase is **structurally impossible** (no shared
module, import, test run, or green/red). To break wildfire you'd have to open the
wildfire repo.

**Rejected:**
- *Branch for oceanography* — anti-pattern; branches aren't project namespaces,
  diverge forever, and force constant switching.
- *Shared-in-place monorepo* — editing the shared `evals/framework` changes
  wildfire's behavior and reds its tests = exactly the maintenance burden to
  avoid. (This is why the earlier "monorepo-now-extract-later" idea is dropped.)
- *Same-repo vendored copy* — gives logical isolation but still shares git
  history + test run + repo green/red, so you'd still feel the pull to fix
  wildfire. Weaker than a clean separate repo; not worth the muddiness.

**Trade accepted:** framework improvements made in oceans won't propagate back to
wildfire (no backport obligation). Fine — drift is acceptable.

**DB:** same Postgres instance (PostGIS + pgvector already enabled), a
`conservation.*` schema — or a separate database. Connection config only; no
wildfire entanglement either way.

**Escalation (probably never):** if you later genuinely want a framework change
shared across both, lift the framework into a local editable-install package both
repos depend on — local dependency hygiene, NOT a PyPI release (which stays out
of scope). Don't do this preemptively.

---

## Build sequencing (decided) — components first, infra last

A correction carried from the wildfire eval: there, too much runway went to the
end-to-end *flow and infrastructure* before the *AI play*, and the eval ended up
feeding the agent from **authored cases, not the live DB** anyway. The principle:
**isolate, exercise, and measure the AI reasoning first; data/infra plumbing is
the LAST dependency.** Feed the system-under-test from cases, not the database.

The design we did is the **map and the contracts** — not a flow to make work
end-to-end. Contracts (`ToolResult`, `Trace`/`TrajectorySpec`, the case model)
are exactly what let each component be built and eval'd in isolation.

**Sequence:**
0. **Thin walking skeleton, fully stubbed** — wire the graph START→answer with
   canned tool returns + a stub answer, just to prove the pieces compose. Reuse
   the `STUB_CODE_INTEL`-style flag. Half a day; no real AI, no DB. (Insurance
   against building components that don't string together — the local-optima
   trap of pure bottom-up.)
1. **Each tool in isolation, eval'd from cases** — most of the AI play lives
   here: `photo_id` (embeddings, LoRA, calibrating the `abstain` threshold)
   scored Layer-A-style; `hybrid_search` tuned on retrieval cases. No agent, no
   graph, no DB-as-source.
2. **Agent orchestration with tools MOCKED from cases** (the `LogisticsTask`
   `mock_tools` pattern) — prompt/ReAct/disambiguation/recovery, scored by
   Layer C, **zero infrastructure**.
3. **Integrate real tools → DB → end-to-end, last** — the smallest, least
   surprising step, because every piece was already proven against cases.

Steps 1–2 are all AI, zero plumbing — the direct fix for the wildfire regret.

---

## Reuse from the current (code-intel) work

- **Eval framework** — Layer-A `RetrievalRanking` scores photo-ID NN matches
  *verbatim* (Precision@k/MRR); Layer-B judge scores the agent's answers.
- **Vector search** — image embeddings instead of text; same retrieval shape.
- **LLM/prompt registries**, LangSmith tracing, the agent/graph patterns.
- **Multimodal thinking** — the gated multimodal scope from the code-intel doc.
- **Oceanography world-domain seam** — env layers (SST, currents) could feed a
  geospatial/risk tool later.

---

## Data (real, public) — for when this starts

- **Photo-ID:** Kaggle *Happywhale* (whale & dolphin ID); *Flukebook* / Wildbook.
- **Knowledge corpus:** NOAA stock-assessment reports; OBIS-SEAMAP sightings.
- **Audio (if ever):** Watkins Marine Mammal Sound DB, DCLDE, NOAA SanctSound;
  pretrained bioacoustic models: Perch/SurfPerch, AVES, BioLingual.
- **Geospatial (stretch):** OBIS-SEAMAP × public AIS (Marine Cadastre) for
  ship-strike risk.

---

## Still open (not yet decided)

- **The "real" bar:** reproduce a published baseline / Happywhale leaderboard
  range, *or* combine datasets into a useful artifact (risk map), *or* a tool a
  hypothetical researcher could use?
- **Output artifact:** agent CLI, notebook + writeup, reproducible eval report?
- **Closed-set vs open-set re-ID:** open-set ("this is a *new* individual") is
  the honest, harder, stronger-eval version. Closed-set is easier. Flagged.
- **Single project vs. sequence** (domain-literacy spike → photo-ID tool →
  agent that wraps it).
- **Corpus sourcing** for the individual-level knowledge content.
- **Cross-cutting training dial:** settled toward *light LoRA on the photo-ID
  embedder only; pretrained everywhere else.*

---

## Next step — data reconnaissance spike (before ANY build)

> **RESOLVED 2026-06-04 — go/no-go answered: GO. The per-individual hub exists in
> open data.** First written as a NO-GO; **overturned** by loading the actual
> OBIS-SEAMAP datasets (see `docs/research/data-research.md` → CORRECTED, and
> `data-build-plan.md`).
>
> **The finding:** OBIS-SEAMAP hosts **Happywhale-contributed datasets** carrying
> `organism_id` (stable per-individual id), `external_resource` (the fluke photo,
> publicly fetchable), and real lat/lon/date — the *entire* identity → history →
> photo bridge in **one namespace**, **openly licensed** (CC0-dominated). Humpback
> alone: ~32k individuals, 20k+ with multi-sighting histories, up to 703 sightings
> on one animal. The platform's front door is gated, but the data flows *out* to
> the public biodiversity repository — the earlier NO-GO generalized from one
> species-occurrence file that happened to lack `organism_id`.
>
> **Consequences:**
> - **The per-individual design stands as originally written** — photo-ID → that
>   individual's sighting history → range → AIS, on **real** data.
> - **Emergent disambiguation (§"showcase of genuine agency") is back** — real
>   look-alikes with real location histories to cross-reference.
> - **No synthetic data, no species-level fallback, no scraping, no data request.**
>   The legitimate open source *is* the hub; reproducible and citable.
> - **Kaggle is now optional** (embedder training / leaderboard only; its
>   anonymized hashes do not join `organism_id`).
>
> Open questions below (real bar, species/region, open-set) resolve toward the
> **real per-individual** project; species = humpback, region = SB Channel or
> Salish Sea (pending a per-region count).

The riskiest assumption in the whole design is that the **individual-level hub
join exists in public data**: photo-ID `individual_id` → that individual's
sighting history (date+location) → range → AIS. If it doesn't join cleanly, the
multi-hop/hub story weakens and the design adapts. Answer this *before* building
anything — and it's hands-on data work, not architecture, so it fits the
components-first lesson.

Questions to resolve:
1. **Happywhale fields** — does it carry per-image/individual **location + date**,
   or just image + `individual_id` + species? (Determines if photo-ID alone
   yields any sighting signal.)
2. **The hub join** — is there public data where photo-ID individuals connect to
   sighting histories? Test Happywhale ↔ OBIS-SEAMAP (likely *fails* — OBIS is
   occurrence data, rarely keyed to named individuals), OR find a single catalog
   that bundles individual + sightings.
3. **NOAA SAR granularity** — individual-level or **stock/species-level**?
   (Likely stock-level → adapt "what's known about this individual" to
   "…this stock," still fine.)
4. **AIS** — confirm Marine Cadastre access + that a density grid is buildable
   for a region/time.
5. **Species/region focus** — does anchoring on ONE species beat multi-species
   Happywhale? Candidate: **North Atlantic right whale** — richly catalogued with
   per-individual sighting histories, and ship-strike/entanglement *is* the
   documented mortality driver, so the AIS multi-hop is the **real** conservation
   question. Check data access (catalog may be gated) vs. Happywhale (open but
   thinner sighting metadata).

**Output of the spike:** a go/no-go on the hub design + any adaptation, which
*also resolves* several "Still open" items — the "real" bar, species/region, and
corpus sourcing. Only after this: walking skeleton → components → integrate.
