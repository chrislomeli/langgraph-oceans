# Tool design — the Layer-3 tools (the agent's hands)

*Status: design in progress (2026-06-19). The actual tools, one per spoke of the hub-and-zoom
model. Each tool has three facets: **Ad** (what the agent is told — see
`agent-orchestration-design.md` for the template + mental model), **Query** (how we retrieve),
and **Storage** (what it stands on). Built on `scope-and-coverage.md` (levels) and
`rag-collections.md` (collections).*

## Method — the query finalizes the storage

We design each tool by writing its **query** against the data. If the query can be written
against storage we have, the storage is validated. Where a query has **no home yet**, that
gap *is* the new-storage spec — the query tells us exactly what columns/indexes it needs. So
this doc is also where the "design the data first" question finally closes: storage is whatever
the queries demand, nothing more.

---

## The tools, by level

### Anchor — `photo_id` (the hub)
- **Ad (condensed):** always first; returns ranked individual(s) + **species** + `abstain`/`margin`; hands `individual_id` to *life*, `species` to *kind*.
- **Query:** cosine kNN over `fluke_embeddings` (HNSW), aggregate top images → individuals; `abstain`/`margin` from the top scores (τ=0.54, calibrated B0).
- **Storage:** `fluke_embeddings` `vector(512)` + join `sightings`(thumb_url) + `individuals`(name, species). **✅ built.**

### Level 1 — its life *(keyed on `individual_id`)*
**`sighting_lookup`**
- **Ad:** where/when this whale has been; its home range. Returns records + `range_bbox`.
- **Query:** `SELECT … FROM sightings WHERE individual_id = ?`; percentile-trimmed `ST_Extent` → core `range_bbox` (PostGIS).
- **Storage:** `sightings` (geography Point) + `individuals`. **✅ built.**

**`sighting_context`**
- **Ad:** what kind of water it uses — depth, SST, sanctuary, ecoregion.
- **Query:** for the individual's sightings, read the `oceano` JSON (depth/SST/sanctuary/region) joined via `source_row_id`.
- **Storage:** `obis_seamap_points.oceano` (JSON, 100% coverage). **✅ built.** *(chlorophyll/salinity also present — regional coverage; un-surfaced extension.)*

### Level 2 — its kind *(keyed on `species`)* — ⚠️ both need NEW storage
**`stock_facts`** *(SAR headline numbers)*
- **Ad:** population status — abundance, PBR, trend, status, strike-vs-entanglement split, + the observed-vs-estimated caveat.
- **Query:** `SELECT * FROM stock_status WHERE species = ?` — a **trivial keyed lookup**.
- **Storage:** **NEW — `stock_status`** `(species PK, abundance, n_min, pbr_us, trend, status, ship_strike_deaths_yr, entanglement_deaths_yr, strike_detection_caveat, source, year)`. ~6 rows, **hand-curated from the 9 SARs.** → *The lookup query proves the "facts→card" decision: a keyed lookup needs a tiny table, not chunks.*

**`doc_search`** *(narrative corpus — `doc_type` is the menu)*
- **Ad:** the *"why"* narrative — `doc_type` ∈ {SAR-reasoning, sanctuary, review}, scoped by species/sanctuary.
- **Query:** **hybrid** — metadata pre-filter (`species` / `sanctuary` / `doc_type`) → vector similarity **+** `tsv` keyword rank → top-k.
- **Storage:** **NEW — `doc_chunks`** `(chunk_id, text, header, species text[], sanctuary, doc_type, section, source, year, embedding vector(D), tsv tsvector)` + `hnsw(embedding)` + `gin(tsv)`. → *The hybrid query fully specifies this table — **every column is determined except `D`, the embedding dimension = the deferred embedder choice.** That decision is now correctly the **last** storage decision, and fully framed: we know everything about `doc_chunks` except D.*
- **Topology *(LOCKED 2026-06-19)*:** **one `doc_search` tool with a `doc_type` filter** — one `doc_chunks` table, one hybrid-retrieval mechanism, one tool to advertise; `doc_type` is the menu within. (Not three separate narrative tools.)

### Level 3 — its waters *(keyed on `range_bbox`)*
**`vessel_traffic`**
- **Ad:** how busy/managed the water is; ship-strike risk only *crossed with* range (life) + species vulnerability (kind).
- **Query:** PostGIS raster — `range_bbox` ∩ `ais_YYYY` rasters → transit metrics; ∩ `vsr_zones` → `lane_overlap`.
- **Storage:** `ais_2022…2025` rasters + `vsr_zones` polygons. **✅ built.**

---

## Storage finalization — what the queries demand

| Tool | Storage | Status |
|---|---|---|
| photo_id | `fluke_embeddings` (+ sightings, individuals) | ✅ exists, validated |
| sighting_lookup | `sightings`, `individuals` | ✅ exists, validated |
| sighting_context | `obis_seamap_points.oceano` | ✅ exists, validated |
| vessel_traffic | `ais_2022…2025`, `vsr_zones` | ✅ exists, validated |
| **stock_facts** | **`stock_status`** — tiny species-keyed table | ⬜ **new (trivial; hand-curate ~6 rows)** |
| **doc_search** | **`doc_chunks`** — metadata + `embedding vector(D)` + `tsv` | ⬜ **new — only free parameter is `D` (embedder)** |

**The whole storage design reduces to two new tables, and the only undecided value in either
is `D`.** So the embedder/dimension question we deferred is the *last* thing standing — and we
deferred it for exactly the right reason: it's downstream of the queries, the schema, and the
chunking. Everything else is built or trivially specified.

## Next
- ✅ `doc_search` topology settled (one tool + `doc_type`). ✅ SAR designed — `stock_status` field list + SAR chunk rules in `sar-collection-design.md`.
- Next collections: #2 Sanctuary Condition → #3 Management → #4 Reviews (each adds its `doc_type` chunk rules; sanctuary/mgmt may add structured fields like the card did).
- `D` (embedder) decided last, when `doc_chunks` is built.
