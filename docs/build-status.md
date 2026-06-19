# Build status — the one dashboard

> **Purpose:** the single place to see *every feature we're building and where it
> stands.* The other docs are deep-dives; this is the map. When they disagree with
> reality, fix this file first. Last verified: **2026-06-18**.
>
> Legend: ✅ done · 🔄 in progress · 🟡 data/deps ready, not built · ⬜ not started · ⚠️ loose end

---

## The product in one line

Show it a whale photo and ask a question. It identifies the individual, reasons about
uncertainty (disambiguates look-alikes, flags new animals), chains across sighting +
shipping data to answer, and cites its sources + shows its reasoning.

> **Scope boundary (2026-06-19):** this is a **conservation-risk reasoning agent, not a
> general whale Q&A bot.** It covers identity · location · conditions · population · threats ·
> protection · ship-risk — and gracefully says "not in my sources" outside that lane. The
> headline answers are *chains across collections*, not single lookups. Full map + the
> in-scope test: `design/scope-and-coverage.md`.

---

## Features (what the user sees) → build status

| # | Feature (user-facing) | Status | Blocked on |
|---|---|---|---|
| **F1** | **"Who is this whale?"** — photo → ranked individual(s) + confidence + catalog match | ✅ **working** | — |
| **F2** | **"Is it a *new* whale?"** — abstains / says NOVEL instead of guessing | ✅ **calibrated** (soft signal) | threshold set on val (B0); overlap is high → agent must treat as a prior, not a verdict |
| **F3** | **"Which one really?"** — disambiguates look-alikes by cross-referencing location/date | ⬜ | needs the agent (B5) + sighting_lookup (B2) |
| **F4** | **"What's known about it?"** — grounded, cited answer from documents | ⬜ | needs text corpus (B3) + hybrid_search (B4) |
| **F5** | **"Is it at risk from ships?"** — range → shipping-traffic overlap (the flagship multi-hop) | 🔄 **chain proven** | tools + scripted chain DONE (B2a+B2b); needs the agent (B5) to make it *agentic* not scripted |
| **F6** | **"Show your work"** — answer + citations + decision trace | ⬜ | needs the agent + trace field (B5) |

**Read this:** F1 is real and good. F5 is the next honest build — all its **data exists**,
it just needs two tools + a chain. F3/F4/F6 need the agent brain and/or the text corpus.

---

## Build items (the technical work) → status

### Phase A — Photo-ID retrieval (the exhibit) — ✅ essentially DONE
| ID | Item | Status | Notes |
|---|---|---|---|
| A1 | Relational + geo data (`individuals`, `sightings`) | ✅ | ~114k sightings / ~23.7k individuals, geo-indexed |
| A2 | Image download → blobs + `manifest` | ✅ | catalog on disk at `image_root` |
| A3 | Image embedding → `fluke_embeddings(512)` | ✅ | **full catalog at v3: 114,089 vectors verified** |
| A4 | Vector index (pgvector HNSW cosine) | ✅ | exact-scan path also exists for eval |
| A5 | `photo_id` tool (kNN + abstain) | ✅ | `src/tools/photo_id.py` + `contracts.py` |
| A6 | **Embedder model** (the ML track) | ✅ **CLOSED** | v3 EffNetV2-S+ArcFace@384, **test reid@1 0.619 / MRR 0.672**; replaced CLIP+LoRA v2 (0.273) |
| A7 | Layer-A eval (retrieval: reid@k / MRR) | ✅ | `reid_split` val/test, `evals/photo_id/eval.py` |

### Phase B — The agent spine (where "agentic" is earned) — mostly ⬜
| ID | Item | Status | Notes |
|---|---|---|---|
| B0 | **Calibrate `abstain` threshold** | ✅ **done** (2026-06-19) | `0.80`→**`0.54`** via Youden's J on reid_split **val** (`evals/photo_id/calibrate_abstain.py`). Genuine/impostor top-1 distributions overlap heavily → balanced acc ~0.70 (false-abstain ~43%, false-match ~16%). NOVEL is a **soft prior**, not a verdict — B5 agent corroborates with `margin` + sighting/location. Unblocks F2 |
| B1 | Tool contracts (`ToolResult`/`Filters`/`Citation`) | ✅ | `src/tools/contracts.py` already built |
| B2a | **`vessel_traffic` tool** | ✅ **built** (2026-06-18) | `src/tools/vessel_traffic.py`; AIS raster → transit metrics + `lane_overlap`. Verified: SB Channel 28.2 vs quiet 1.2 mean/cell (23×). Query now committed |
| B2b | **`sighting_lookup` tool** | ✅ **built** (2026-06-18) | `src/tools/sighting_lookup.py`; sightings → records + **core `range_bbox`** (percentile-trimmed so migratory outliers don't make a continent-box). **F5 chain proven end-to-end** on whale 479: Monterey core range → 2.4M transits/yr inside Monterey Bay NMS |
| B3-PRE | **RAG preprocessing** (chunk + annotate) | 🔄 **design done** | section-aware chunks + **join-key metadata** (species ↔ photo_id, sanctuary ↔ vessel_traffic) + contextual headers; the annotation is the lever (bare chunks mix stocks). Design: `design/rag-preprocessing-design.md` (+ per-collection designs in `design/`). Impl pending |
| B3 | `doc_chunks` ingestion (embed + load) | ⬜ | 21 PDFs **acquired** (SAR/sanctuary/reviews in `~/Source/DATA/oceans/corpus/`); run B3-PRE → text-embed → load. Checkpoint: a scoped "humpback mortality" query returns humpback-only chunks |
| B3-OBIS | OBIS-density tool (structured) | 🟡 own-data | separate track from the corpus; the disambiguation prior |
| B-ENV | **`sighting_context` tool** (oceano enrichment) | ✅ **built** (2026-06-18) | `src/tools/sighting_context.py`; depth / SST / sanctuary / region per individual from the OBIS `oceano` JSON (already in `obis_seamap_points`, 100% coverage, joins via `source_row_id`). **Un-parks bathymetry + SST — no GEBCO/ERDDAP needed.** Depth/shelf-fraction = the F5 risk-modulation lever |
| B4 | `hybrid_search` / `catalog_search` tools | ⬜ | depends on B3 |
| B5 | Agent graph (router → ReAct → recovery) | ⬜ | greenfield; LangGraph. The actual "agency". Trace must be a first-class field |
| B5-CTX | **Context layer** (Layer 2.5: state · assembly · provenance · trace; later compression/budget/fusion) | ⬜ **extract from the vertical** | framework-grade, infra-flavored. MVP = state+assembly+provenance+trace, lifted out of the ship-strike agent once the scratchpad sprawl is real — **build after B5, not before**. Design: `research/context-layer-design.md` |
| B6 | Grounded synthesis (LLM + citations) | ⬜ | the synth node |
| B7 | Layer-B eval (answer faithfulness) | ⬜ | reuses framework judge |
| B8 | Layer-C eval (orchestration / trajectory) | ⬜ | the "agentic = a number" payoff; local-first (LangSmith quota burned) |

### Phase C — ML lift (the personal learning goal) — ✅ folded into A6
| ID | Item | Status | Notes |
|---|---|---|---|
| C1 | LoRA fine-tune embedder | ✅ | done as the CLIP+LoRA line (v1→v2), then superseded by v3 ArcFace. Learning goal met |

---

## Data substrate → status (what the tools stand on)

| Table / asset | Feeds | Status |
|---|---|---|
| `individuals`, `sightings` (geo Point) | F1, F3, F5 | ✅ loaded |
| `fluke_embeddings` (v3, 512-d) | F1 | ✅ full catalog |
| `ais_2022…2025` (PostGIS raster, transit counts) | F5 | ✅ loaded + **queryable via `vessel_traffic`** |
| NMS / VSR speed-zone polygons (`vsr_zones`) | F5 lane-overlap | ✅ loaded (7 NMS sanctuaries / 12 polygons) |
| `obis_seamap_points.oceano` (depth/SST/salinity/zone JSON) | env context (F5 modulation) | ✅ 100% coverage, via `sighting_context` |
| `obis_seamap_points` remarks (occurrence/organism) | F4 per-encounter text | 🟡 43–54% coverage; modest per-sighting prose, **unused** — a real if small per-individual text source |
| `doc_chunks` (text corpus) | F4 | ⬜ not created (PDFs acquired; ingestion pending) |
| catalog image blobs | A3, F1 | ✅ on disk |

---

## Phase B data sources (F4 RAG + structured context)

The rule that sorts these: **text-RAG only where the answer is genuinely *narrative*
(why it matters, what's being done, status-in-prose); structured for every *fact*
(where, when, how deep, how busy).** Each source is on the list because it **drives a
decision** in an agentic trace — not because it's available. Acquisition spec +
exact documents → `design/rag-corpus-shopping-list.md`.

**Priority 1 — text-RAG corpus (`doc_chunks`), ranked.** Honest scope: stock/region/
threat-level, not per-individual. A curated ~dozens-of-docs corpus, not a scrape.

| # | Source | Agentic role (what the result *triggers*) | Scenario | Status |
|---|---|---|---|---|
| 1 | **NOAA Stock Assessment Reports (SARs)** | reads stock status → **escalate or pivot** the investigation | Severity / conflict | ⬜ to acquire |
| 2 | **Sanctuary Condition Reports + Mgmt Plans** | "is the risk *managed*?" → conclusion flips on the text; triggers Whale Safe hop | Recovery / mitigation | ⬜ to acquire |
| 3 | **Whale Safe methodology + report cards** | "zone exists — are vessels *complying*?" → dependent follow-up | Multi-hop dependency | ⬜ to acquire |
| 4 | **Ship-strike / entanglement reviews** | *mechanism* depth (why humpbacks are vulnerable) | Enrichment | ⬜ **bound to 2–3 open-access** |

**Priority 2 — structured context tools** (low learning value → must be cheap × high payoff):

> **2026-06-18 finding:** the OBIS `oceano` JSON (in `obis_seamap_points`, 100% coverage)
> already carries **depth, SST, salinity, marine ecoregion, and sanctuary per sighting** —
> the enrichment we'd planned to go acquire was in the source all along. Surfaced via the
> `sighting_context` tool (B-ENV). This un-parks/un-defers most of the table below.

| Source | Agentic role | Cost | Verdict |
|---|---|---|---|
| **OBIS density** (own data / OBIS API) | disambiguation **prior** + recovery plausibility | ~free (aggregate) | ✅ **in** — agency. ⚠️ verify it discriminates across candidates before relying on it |
| **Bathymetry (depth)** | risk-modulation + habitat-depth disambiguation | **~free** — already in `oceano` | ✅ **DONE via `sighting_context`** — no GEBCO download needed |
| **SST** | "why was it here" (cool upwelling water) | **~free** — already in `oceano` | ✅ **available via `sighting_context`** — no ERDDAP API needed (chlorophyll not in oceano; SST is) |
| Sanctuary/region (WDPA, LME) | which sanctuary / ecoregion a sighting sits in | ~free — already in `oceano` | ✅ in `sighting_context`; complements the `vsr_zones` geometry |

> Two build tracks, kept distinct: the four text sources = one job (chunk → text-embed →
> `doc_chunks` → `hybrid_search`). **OBIS density is a separate structured tool** (like
> `vessel_traffic`), not part of the corpus.

---

## What's next (the recommended path)

1. ~~**B0** — calibrate the abstain threshold~~ ✅ **done 2026-06-19** (τ=0.54 on val; F2 unblocked).
2. **B2 + F5** — build `sighting_lookup` + `vessel_traffic`, prove the ship-strike
   multi-hop as a **scripted chain over real data** (no AI yet). The honest flagship.
3. **B5** — wrap it in the LangGraph agent so it branches on close candidates
   (disambiguation = F3), turning the chain into genuine agency. **Then extract the
   context layer (B5-CTX)** from the pain it generates — don't build it first.
4. Then B3/B4 (text corpus + RAG) for F4, and B7/B8 (eval) to make "agentic" a number.
   Compression/budgeting/fusion get added to the context layer when RAG floods the window.

> Deep-dives: product/architecture = `ocean-mammal-conservation-vision.md` ·
> embedder story = `research/lora-experiment-log.md` · data = `research/data-build-plan.md`.
> Those are history/detail; **this file is the live status.**
