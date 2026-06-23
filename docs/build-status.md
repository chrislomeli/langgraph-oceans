# Build status — the one dashboard

> **Purpose:** the single place to see *every feature we're building and where it
> stands.* The other docs are deep-dives; this is the map. When they disagree with
> reality, fix this file first. Last verified: **2026-06-20**.
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
| **F5** | **"Is it at risk from ships?"** — range → shipping-traffic overlap (the flagship multi-hop) | ✅ **AGENTIC** (2026-06-21) | the agent now decomposes + chains it ITSELF (photo_id → sighting_lookup ∥ sighting_context → vessel_traffic → synthesis), no script. Verified live on #479 (Monterey humpback → 2.4M transits, Monterey Bay NMS) via `agent.cli` |
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

**Layer-3 tool roster — the delivery list. Definition of done = (a) tool built? + (b) data loaded?**
This is the canonical "every tool we will deliver" list; the lettered build-items below carry the detail.

| Tool | What it lets the agent DO | (a) Built? | (b) Data loaded? |
|---|---|---|---|
| `photo_id` | photo → ranked individual(s) + confidence + abstain/NOVEL | ✅ yes | ✅ yes — `fluke_embeddings` (114k v3), `individuals`, `sightings` |
| `sighting_lookup` | individual → sighting history + core `range_bbox` (feeds vessel_traffic) | ✅ yes | ✅ yes — `sightings`, `individuals` |
| `sighting_context` | individual → depth / SST / sanctuary / region per sighting | ✅ yes | ✅ yes — `obis_seamap_points.oceano` (100%) |
| `vessel_traffic` | `range_bbox` → transit metrics + lane_overlap + sanctuary names | ✅ yes | ✅ yes — `ais_2022–2025` rasters, `vsr_zones` |
| `stock_facts` | species → keyed card (abundance, PBR, strike/entangle deaths, status) | ⬜ no | ⬜ no — `stock_status` table exists but **0 rows** (curate ~6 from SARs) |
| `doc_search` | query + species/sanctuary/doc_type filter → top-k narrative chunks (hybrid) | 🔄 partial — repo `search`/`search_hybrid` built+verified; Layer-3 wrapper only sketched | 🔄 partial — `doc_chunks` SAR (378) loaded; sanctuary (10) + reviews (2) acquired-not-ingested |
| `obis_density` *(optional)* | disambiguation prior — candidate plausibility by recorded density | ⬜ no | 🟡 derivable from `sightings` (own data) — speculative, separate track |

**Read:** 4 of 7 are fully done (built **and** data) — and those 4 are exactly the F5-chain set = the **B5 start gate**. `stock_facts` needs both halves; `doc_search` needs its tool wrapper + the rest of the corpus; `obis_density` is optional. No other tools are planned — this is the whole repertoire.

| ID | Item | Status | Notes |
|---|---|---|---|
| B0 | **Calibrate `abstain` threshold** | ✅ **done** (2026-06-19) | `0.80`→**`0.54`** via Youden's J on reid_split **val** (`evals/photo_id/calibrate_abstain.py`). Genuine/impostor top-1 distributions overlap heavily → balanced acc ~0.70 (false-abstain ~43%, false-match ~16%). NOVEL is a **soft prior**, not a verdict — B5 agent corroborates with `margin` + sighting/location. Unblocks F2 |
| B1 | Tool contracts (`ToolResult`/`Filters`/`Citation`) | ✅ | `src/tools/contracts.py` already built |
| B2a | **`vessel_traffic` tool** | ✅ **built** (2026-06-18) | `src/tools/vessel_traffic.py`; AIS raster → transit metrics + `lane_overlap`. Verified: SB Channel 28.2 vs quiet 1.2 mean/cell (23×). Query now committed |
| B2b | **`sighting_lookup` tool** | ✅ **built** (2026-06-18) | `src/tools/sighting_lookup.py`; sightings → records + **core `range_bbox`** (percentile-trimmed so migratory outliers don't make a continent-box). **F5 chain proven end-to-end** on whale 479: Monterey core range → 2.4M transits/yr inside Monterey Bay NMS |
| B3-PRE | **RAG preprocessing** (chunk + annotate) | ✅ **built** (2026-06-20) | `src/rag/extract_sar.py`: heading-split (case-insensitive + TOC strip, generalized across all 8 SARs) → section-aware chunks + **join-key metadata** (species[] ↔ photo_id, sanctuary ↔ vessel_traffic) + contextual header embedded (not stored in `text`). Design: `design/rag-preprocessing-design.md`. **SAR collection only**; sanctuary/review collections still to chunk |
| B3 | `doc_chunks` ingestion (embed + load) | ✅ **SAR done** (2026-06-20) | **378 SAR chunks loaded**, openai-3-large@1536, idempotent upsert on deterministic `chunk_id` (`ON CONFLICT`). Checkpoint **passed**: scoped "humpback mortality" returns humpback-only Mortality chunks (`src/rag/retrieval_smoke.py`); without the species filter a gray-whale chunk leaks — proving the scoping is load-bearing. Remaining 13 PDFs (sanctuary/reviews) not yet ingested |
| B3-SANCT | **Sanctuary collection** chunk + load (condition + mgmt) | 🟡 **acquired, not built** | 10 PDFs in `corpus/sanctuary/` (5 condition + 5 mgmt). Reuses the shared embedder/repo; needs a collection extractor — problem is **selection** (CR ~90% out-of-lane; mgmt heterogeneous 4pp–479pp, per-doc page ranges) + `sanctuary` join-key populated. Designs: `design/sanctuary-condition-collection-design.md`, `design/sanctuary-mgmt-collection-design.md` |
| B3-REV | **Reviews collection** chunk + load (ship-strike / entanglement) | 🟡 **acquired, not built** | 2 PDFs in `corpus/reviews/` (conn-silber-2013, rockwood-2017). Reuses shared backend; **multi-species `species[]`**, join axis = the *mechanism question* not species; keep Abstract+Discussion, drop Methods/Refs. Design: `design/reviews-collection-design.md` |
| B3-OBIS | OBIS-density tool (structured) | 🟡 own-data | separate track from the corpus; the disambiguation prior |
| B-ENV | **`sighting_context` tool** (oceano enrichment) | ✅ **built** (2026-06-18) | `src/tools/sighting_context.py`; depth / SST / sanctuary / region per individual from the OBIS `oceano` JSON (already in `obis_seamap_points`, 100% coverage, joins via `source_row_id`). **Un-parks bathymetry + SST — no GEBCO/ERDDAP needed.** Depth/shelf-fraction = the F5 risk-modulation lever |
| B-FACTS | **`stock_facts` tool** + `stock_status` curation | ⬜ **untracked until now** | F4's *facts* leg: trivial keyed lookup `WHERE species=?`. `datasets.stock_status` table **exists but 0 rows** — hand-curate ~6 from the SARs (`design/sar-collection-design.md` field list). NOT a B5 gate; the agent acquires it incrementally |
| B4 | `hybrid_search` / `catalog_search` tools | 🔄 **repo layer built** | `DocumentRepository.search` (vector + metadata pre-filter) + `search_hybrid` (RRF fuse vector+`tsv`) built & A/B-verified (`retrieval_smoke.py`); `doc_search` Layer-3 tool wrapping them only SKETCHED (full-text-vs-snippet + query-shaping for tsv AND-semantics still to decide) |
| **B-BIND** | **LLM tool-binding layer** (the B5 gate) | 🔄 **scaffolded 2026-06-20; ads = `TODO(you)`** | **`src/agent/tools.py`** — 4 tools wrapped as LangChain `@tool`, registered & smoke-verified (`uv run python -m agent.tools`). Plumbing done (lazy singletons, F5-chain seam). **YOUR step (tier 2): write each docstring = the tool *ad*** (Trigger / Anti-overlap / Returns, per `design/agent-orchestration-design.md`). The single most important prerequisite to B5 |
| **B-CLI** | **Agent entrypoint / CLI driver** | ✅ **built 2026-06-20** | **`src/agent/cli.py`** — `uv run python -m agent.cli "<q>" [--image PATH] [--trace]`; invokes the graph, renders answer + optional tool-call trace. Single-shot (multi-turn chat deferred *with* B5-CTX — same checkpointer) |
| ▶ **GATE** | **Minimum to START B5** | — | **B-BIND + B-CLI + pick agent LLM (Opus 4.8)**. That's it. The 4 structured tools are built; the F5 chain needs **zero corpus**. doc_search/stock_facts/collections/evals are *parallel-or-after*, NOT prerequisites |
| B5 | Agent graph (router → ReAct → recovery) | 🔄 **RUNS END-TO-END** (2026-06-21) | **`src/agent/graph.py`** — ReAct loop over the 4 bound tools; **F5 chain runs agentically** (see F5). Ads + system prompt written (starters, Claude-owned per agreement). Remaining: router/recovery nodes (deferred), eval-driven tuning (B8), prompt caching. Design: `design/agent-graph-design.md`. Model `claude-opus-4-8` (temperature omitted — deprecated for this model) |
| B5-CTX | **Context layer** (Layer 2.5: state · assembly · provenance · trace; later compression/budget/fusion) | ⬜ **extract from the vertical** | framework-grade, infra-flavored. MVP = state+assembly+provenance+trace, lifted out of the ship-strike agent once the scratchpad sprawl is real — **build after B5, not before**. Design: `research/context-layer-design.md` |
| B6 | Grounded synthesis (LLM + citations) | ⬜ | the synth node |
| B7 | Layer-B eval (answer faithfulness) | ⬜ | reuses framework judge. **Hidden work: authoring the Q + grounded-answer cases** (the harness is reused; the dataset is not) |
| B8 | Layer-C eval (orchestration / trajectory) | ⬜ | the "agentic = a number" payoff; local-first (LangSmith quota burned). **Hidden work: authoring scenario cases** (question → expected trajectory) — historically under-estimated |

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
| 1 | **NOAA Stock Assessment Reports (SARs)** | reads stock status → **escalate or pivot** the investigation | Severity / conflict | ✅ **ingested** (9 in `corpus/sar/`; 8 chunked → 378 `doc_chunks`, B3) |
| 2 | **Sanctuary Condition Reports + Mgmt Plans** | "is the risk *managed*?" → conclusion flips on the text; triggers Whale Safe hop | Recovery / mitigation | 🟡 **acquired, ingestion pending** (10 in `corpus/sanctuary/`: 5 condition + 5 mgmt) |
| 3 | **Whale Safe methodology + report cards** | "zone exists — are vessels *complying*?" → dependent follow-up | Multi-hop dependency | ⬜ to acquire (no `corpus/whalesafe/` yet) |
| 4 | **Ship-strike / entanglement reviews** | *mechanism* depth (why humpbacks are vulnerable) | Enrichment | 🟡 **acquired, ingestion pending** (2 in `corpus/reviews/`: conn-silber-2013, rockwood-2017) |

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
