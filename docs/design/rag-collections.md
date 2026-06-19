# RAG collections — the working list

*Status: working list (2026-06-19). The stable surface we go through **one collection at
a time** to design retrieval. Companion to `rag-corpus-shopping-list.md` (what to acquire +
the demo questions) and `rag-preprocessing-design.md` (chunk + annotate strategy). The
embedder choice is deliberately deferred — it only pins the vector dimension and is
re-doable via `embedder_ver`; design the retrieval contract first.*

## What counts as a "collection"

A **collection = a distinct body of knowledge that needs its own design treatment** — its
own chunk/annotate rules (if text) or its own extraction (if structured). The test for
"separate collection": does it have a **different internal structure** *and* answer a
**different class of question**? Bodies that are already solved as tools are out of scope
for this pass (see below).

The design conversation runs **per collection**. For each we'll settle: what the agent is
told it can ask of it (the menu), what questions it answers, how it's chunked + annotated,
images (drop/blob/file), and whether any of its content is really structured-fact that
belongs in a table+tool rather than in chunks.

---

## Text collections — need chunk/annotate design

| # | Collection | On disk | Count | Why it's its own design unit | Designed? |
|---|---|---|---|---|---|
| 1 | **SAR** — NOAA Stock Assessment Reports | ✅ `corpus/sar/` | 9 | Rigid section schema (Stock Definition · Pop Size · Trend · PBR · Human-Caused Mortality · Status); the backbone. The "facts buried in mangled tables" problem lives here. | ✅ `sar-collection-design.md` |
| 2 | **Sanctuary Condition Reports** | ✅ `corpus/sanctuary/` | 5 | Narrative, figure-heavy (20–37 MB). Answers *"what pressures / is the risk acknowledged?"* | ✅ `sanctuary-condition-collection-design.md` |
| 3 | **Sanctuary Management Plans** | ✅ `corpus/sanctuary/` | 5 | Action-oriented. Answers *"what's being done?"* | ✅ `sanctuary-mgmt-collection-design.md` |
| 4 | **Reviews** — ship-strike / entanglement papers | ✅ `corpus/reviews/` | 2 | Academic structure (Abstract/Results/Discussion); species-*agnostic* mechanism depth. Capped at 2. | ✅ `reviews-collection-design.md` |

**Opinion #1 — split sanctuary into two collections (#2, #3).** The shopping list lumped
Condition Reports + Management Plans under one "sanctuary" bucket, but they're structurally
different and answer different questions, and they split clean on disk (5 + 5). One shared
chunking rule would fit neither. → **4 acquired text collections, not 3.**

**Filed under SAR, not separate:** the SAR-numbers-as-structured question (abundance, PBR,
annual ship-strike vs entanglement mortality, trend, status — currently table-bound facts).
Whether those become a small `stock_status` table + tool vs. stay as linearized chunks is a
**transform of collection #1**, decided when we design SAR — not a new collection. (Open.)

---

## Structured collections — kept in view this conversation

Not text-RAG, but in scope for the design (how their *facts* enter the agent's answer, and
what in them is still unexploited).

| # | Collection | Served by | Design status / what's unexploited | Designed? |
|---|---|---|---|---|
| 5 | **AIS vessel-traffic density rasters** — `ais_2022 / 2023 / 2024 / 2025` (+ `ais_all`) | `vessel_traffic` | Per-year transit-count density grids for *all* shipping. The "protected" view = traffic ∩ `vsr_zones` (NMS/VSR polygons) via `lane_overlap` — **two layers meeting in the tool**, not a single "protected lanes" dataset. Retrieval is solved; open question is only *how AIS facts surface in the narrative answer*. ⚠️ if "protected lanes" meant published TSS/lane geometries, that's a 3rd layer we don't have. | ✅ tool |
| 6 | **`obis_seamap_points`** — the OBIS-Happywhale **source staging table** | partly: `sightings` (relational), `sighting_context` (`oceano`) | The richest raw source; most is exploited, but **three slices are not** (below). The home for old edge (B). | 🔄 partial |

**Collection 6 — the unexploited slices (from the 2026-06-19 record sample):**

- **`oceano` carries more than `sighting_context` surfaces.** Confirmed present in the JSON:
  `CHLOR-A` (MODIS chlorophyll — *overturns the earlier "chlorophyll not in oceano" note*),
  `SALINITY` (HYCOM), **multi-source SST** (OISST/MODIS/HYCOM), and richer `ZONE`
  (EEZ/LME/MEOW/**WDPA** sanctuary). ⚠️ **coverage looks regional** — the Mexico record had
  neither chlorophyll nor salinity; both Hawaii records had both. Verify coverage before
  relying.
- **Per-encounter remarks (`occurrence_remarks`) — weaker than it sounds.** Multilingual,
  mostly field-logistics boilerplate (*"Tamaño grupo: 2 (par)…"*, *"(no remarks)"*,
  *"HIHWNMS ID… Courtesy E. Lyman…"*). Honest downgrade of old edge (B): this is **not**
  narrative the agent reasons over.
- **`organism_remarks` hides an identity signal, not prose.** e.g. *"Also formerly
  RCHP-11RUCO868 and RCHP-13RUCO1338"* — **alternate catalog IDs for the same animal**.
  That's a **disambiguation / identity-linking** fact (a whale seen under multiple catalog
  names), categorically different from RAG.

---

## Resolved edges

- **(A) Whale Safe — TABLED (2026-06-19).** No data found that made sense; revisit only if a
  gap surfaces. Removed from scope.
- **(B) OBIS per-encounter remarks — kept, folded into Collection 6.** Reclassified after
  looking at the data: the prose is low-value boilerplate, but the `oceano` extras and the
  `organism_remarks` alt-IDs are worth carrying. Not its own collection.

---

## Status

The list is **closed for design**. Designed so far:
- **#1 SAR** (`sar-collection-design.md`) — card (`stock_status`) + chunks; the card-vs-chunks split.
- **#2 Sanctuary Condition** (`sanctuary-condition-collection-design.md`) — **no card**, chunks only;
  dominant problem is *selection* from a 482-pp doc (hand-map page ranges); join key = `sanctuary`;
  CR owns *pressure/severity*, Mgmt owns *mitigation* (the seam).
- **#3 Sanctuary Management** (`sanctuary-mgmt-collection-design.md`) — **no card**; owns the
  *mitigation* side of the seam; the 5 files are **heterogeneous** (4-pp memo → 479-pp plan) so
  mechanics are *per-doc, shape-specific*; known gap: Monterey 2008 predates modern VSR.
- **#4 Reviews** (`reviews-collection-design.md`) — **no card**; join axis = the *mechanistic
  question* (not species/sanctuary); keep Abstract+Discussion only; 2 docs, trivial mechanics.

**ALL 4 TEXT COLLECTIONS DESIGNED.** Remaining before build: decide **D** (the embedder
dimension) → build `doc_chunks` + `stock_status`. Structured collections #5 (AIS) / #6
(obis_seamap_points) handled in their tool work, not here.
