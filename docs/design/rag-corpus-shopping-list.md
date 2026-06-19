# RAG corpus — shopping list & acquisition spec

> **What this is:** the concrete documents to acquire for the F4 knowledge layer, *why*
> each is on the list (the agentic decision it drives), and the demo questions the corpus
> must answer (its definition of done). **You acquire the source files**; once they're on
> disk, the build job chunks + text-embeds them into `datasets.doc_chunks`.
>
> **Scope:** the **6 Eastern North Pacific species actually in the DB** — Humpback (108k),
> Killer Whale (2.6k), Gray (2.3k), Blue (404), Fin (110), Sperm (65) — in their correct
> ENP / CA-OR-WA stock. (Originally scoped humpback-only; the DB species breakdown upgraded
> it to multi-species, so the agent can answer about any sighting it IDs.) Matches your AIS +
> sanctuary data (Channel Islands & Chumash Heritage NMS, SB Channel). Theme: **ship-strike
> risk** — F4 is the "why it matters / what's being done" voice over the F5 flagship.
>
> **Honest scope reminder:** these are **stock / region / threat-level**, never per-animal.
> The corpus answers "what's known about this whale's *population and the threats it faces*."
>
> **URLs are entry points** — grab the *current* version from the official site; don't trust
> a stale link. Prefer open-license / public-domain (NOAA is public domain; pick open-access
> papers).

---

## The modality split (why some questions are NOT in the corpus)

Text-RAG answers the **narrative** ("why / what's being done / status-in-prose"). **Facts**
(where / when / how deep / how busy / is-it-in-a-zone / usually-here?) are answered by
**structured tools**, not the corpus. Each demo question below is tagged `[RAG]` or
`[TOOL]` so we don't embed data that a column answers better.

---

## Priority 1 — the text corpus (4 sources)

### 1. NOAA Stock Assessment Reports (SARs) — *the backbone* ✅ ACQUIRED
- **Status:** DONE. 9 curated PDFs (the 6 DB species in their ENP/CA-OR-WA stock + the
  combined 2024 Pacific SAR) are in **`~/Source/DATA/oceans/corpus/sar/`** with a manifest
  README. Curated from a larger `~/Downloads/SAR` collection; species not in the DB and
  wrong-region stocks were excluded.
- **Agentic payload confirmed by reading them:** each SAR carries the **vessel-strike vs
  entanglement mortality split**, so the F5 severity branch **pivots per species** (humpback =
  entanglement-dominant, vessel strike minor; blue/fin = vessel strike is a top threat).
- **Format:** PDF · **License:** public domain (NOAA).
- **Drives:** the **severity branch** — stock status decides whether the agent escalates
  (small/declining, strike = top threat) or pivots (recovering, threat is entanglement).
- **Answers:** `[RAG]` population status/trend · abundance estimate · PBR · annual human-caused
  mortality (ship strike vs entanglement) · strategic-stock status.

### 2. Sanctuary Condition Reports + Management Plans — *the mitigation branch* ✅ ACQUIRED
- **Status:** DONE. 10 curated PDFs in **`~/Source/DATA/oceans/corpus/sanctuary/`** (manifest
  README) — the 4 California sanctuaries (Channel Islands = primary/SB Channel, Greater
  Farallones, Cordell Bank, Monterey Bay) + Olympic Coast (WA), each with its current
  Condition Report + Management Plan. Other coasts, historical EIS volumes, and superseded
  editions excluded. Channel Islands CR verified to carry heavy ship-strike/vessel content.
- **Ingestion note:** Condition Reports are large (20–37 MB, figure-heavy) — extract **text
  only**, don't embed the images.
- **Format:** PDF · **License:** public domain (NOAA).
- **Drives:** the **recovery/mitigation branch** — once F5 finds traffic overlap, the plan
  says whether a measure exists (VSR / voluntary 10-kt zone). The conclusion flips on this text.
- **Answers:** `[RAG]` pressures on the sanctuary (vessel traffic, ship strike) · management
  actions in force · ecological setting · how the VSR program works.

### 3. Whale Safe — methodology + report cards — *the compliance hop*
- **Get:** the **methodology** page/whitepaper, and the **vessel-cooperation report cards**
  for the SB Channel and SF Bay Area VSR zones.
- **Where:** `whalesafe.com` (Benioff Ocean Science Initiative, UCSB + partners).
- **Format:** web pages + downloadable report-card PDFs/data · **Effort:** low · **License:**
  check page terms (cite as reference; don't claim to reproduce the system).
- **Drives:** the **text multi-hop** — after the mgmt plan says "a zone exists," the agent asks
  "are vessels actually complying?" and pulls the report card. Retrieval N+1 built from N.
- **Answers:** `[RAG]` how strike risk is scored · vessel-cooperation rates with the slowdown.

### 4. Ship-strike / entanglement reviews — *mechanism depth (bounded!)* ✅ ACQUIRED
- **Status:** DONE. 2 open-access papers in **`~/Source/DATA/oceans/corpus/reviews/`**
  (manifest README): **Rockwood et al. 2017** (West Coast blue/humpback/fin mortality — on
  region + 3 DB species) and **Conn & Silber 2013** (speed→lethality mechanism). **Capped at
  2 — do not add more** (this is the weakest-agency garnish source).
- **Format:** PDF · **License:** open-access.
- **Drives:** **enrichment** (honest: weakest agency) — the "why humpbacks are vulnerable /
  how speed affects lethality" depth. The best justification for *real* semantic search (messy
  prose). Keep it small.
- **Answers:** `[RAG]` why humpbacks are strike-prone · does speed change lethality · regional
  mortality estimates.

---

## Priority 2 — structured context (NOT corpus; acquire-if-building)

### OBIS density — the disambiguation prior `[TOOL]`
- **You may not need a download:** your `datasets.sightings` is already OBIS-Happywhale-derived,
  so density-by-quadrant-by-month can be aggregated from data you have. For broader multi-species
  context, pull occurrences via the **OBIS API** (`api.obis.org/v3`) or the **OBIS GeoParquet on
  AWS**.
- **Drives:** disambiguation prior ("is this candidate plausible *here, this month*?") + recovery
  plausibility. ⚠️ **Verify it discriminates across candidates before relying on it** (humpbacks
  cluster; the prior may not separate look-alikes).
- **Answers:** `[TOOL]` are these whales usually found here this time of year?

### GEBCO bathymetry — PARKED stand-in `[TOOL]`
- **Get (only if/when a demo needs the visual):** **GEBCO 2024 Grid**, subset to the CA Current
  region (the global grid is large). `gebco.net` → Data Download → GEBCO_2024 Grid.
- **Format:** NetCDF / GeoTIFF → load with `raster2pgsql` exactly like the AIS rasters.
- **Status:** not in the build path — the cheapest demo-wow visual (sightings over seafloor
  canyons), reuses your `vessel_traffic` raster code. Decide later.

---

## Demo questions — the corpus's definition of done

The corpus is "good enough" when it can ground these. (Tagged by modality so we don't embed
what a tool answers.)

| Demo question | Source | Modality |
|---|---|---|
| What's the population status / trend of this whale's stock? | SAR | `[RAG]` |
| How many of these whales are estimated to exist? | SAR | `[RAG]` |
| What are the main human-caused threats to this population? | SAR | `[RAG]` |
| How many are killed by ship strikes per year on the West Coast? | SAR / Rockwood 2017 | `[RAG]` |
| Why are humpbacks vulnerable to vessel strikes? | reviews | `[RAG]` |
| Does slowing ships down actually reduce lethal strikes? | Conn & Silber / reviews | `[RAG]` |
| What's being done to reduce strikes in the Santa Barbara Channel? | Sanctuary mgmt plan + Whale Safe | `[RAG]` |
| Are ships actually slowing down in this zone? | Whale Safe report cards | `[RAG]` |
| Is this sighting inside a protected sanctuary? | `vsr_zones` | `[TOOL]` |
| How much ship traffic crosses this whale's range? | `vessel_traffic` | `[TOOL]` |
| Are humpbacks typically found here this time of year? | OBIS density | `[TOOL]` |

---

## Acquisition checklist (what you actually do)

- [x] **SARs** — DONE: 9 curated PDFs (6 ENP species + combined 2024) in `~/Source/DATA/oceans/corpus/sar/` (see its README).
- [x] **Sanctuary** — DONE: 10 curated PDFs (4 CA sanctuaries + Olympic Coast) in `~/Source/DATA/oceans/corpus/sanctuary/`.
- [ ] **Whale Safe** — save the methodology page + the SB Channel / SF report cards.
- [x] **Reviews** — DONE: Rockwood 2017 + Conn & Silber 2013 in `~/Source/DATA/oceans/corpus/reviews/`. Capped at 2.
- [ ] **OBIS density** — decide: aggregate existing `sightings`, or pull OBIS API. *(separate tool track)*
- [ ] *(parked)* GEBCO — skip unless a demo needs the visual.

Drop the acquired files somewhere stable (e.g. `~/Source/DATA/oceans/corpus/<source>/`); the
B3 ingestion job will chunk + embed them into `doc_chunks`. Then F4 has a target it can hit.
