# RAG corpus — shopping list & acquisition spec

> **What this is:** the concrete documents to acquire for the F4 knowledge layer, *why*
> each is on the list (the agentic decision it drives), and the demo questions the corpus
> must answer (its definition of done). **You acquire the source files**; once they're on
> disk, the build job chunks + text-embeds them into `datasets.doc_chunks`.
>
> **Scope:** humpback whale, **Eastern North Pacific / California Current** (matches your
> AIS + sanctuary data: Channel Islands & Chumash Heritage NMS, SB Channel). Theme:
> **ship-strike risk** — so F4 is the "why it matters / what's being done" voice over the
> F5 flagship.
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

### 1. NOAA Stock Assessment Reports (SARs) — *the backbone*
- **Get:** *"Humpback Whale (Megaptera novaeangliae): California/Oregon/Washington Stock"* —
  the current Pacific marine-mammal SAR (annual). If in scope, also the **species/DPS status
  review** (2016 ESA rule split humpbacks into DPSs: Central America = endangered, Mexico =
  threatened, Hawaii = not listed — the CA/OR/WA *stock* mixes them; that tension is itself a
  great severity/conflict payload).
- **Where:** NOAA Fisheries → *Marine Mammal Stock Assessment Reports* → Pacific region
  (`fisheries.noaa.gov/national/marine-mammal-protection/marine-mammal-stock-assessment-reports`).
- **Format:** PDF · **Effort:** low (a few bounded, authoritative files) · **License:** public domain.
- **Drives:** the **severity branch** — stock status decides whether the agent escalates
  (small/declining, strike = top threat) or pivots (recovering, threat is entanglement).
- **Answers:** `[RAG]` population status/trend · abundance estimate · PBR · annual human-caused
  mortality (ship strike vs entanglement) · strategic-stock status.

### 2. Sanctuary Condition Reports + Management Plans — *the mitigation branch*
- **Get:** **Channel Islands NMS** Condition Report **and** Management Plan. Add **Greater
  Farallones / Cordell Bank NMS** if SF Bay Area is in scope; **Chumash Heritage NMS**
  designation / draft management documents (designated 2024).
- **Where:** `sanctuaries.noaa.gov` → each sanctuary → *Condition Report* and *Management Plan*
  (Channel Islands: `channelislands.noaa.gov`).
- **Format:** PDF (large; few) · **Effort:** medium · **License:** public domain.
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

### 4. Ship-strike / entanglement reviews — *mechanism depth (bounded!)*
- **Get (2–3, prefer open-access):**
  - **Rockwood, Calambokidis & Jahncke 2017, PLOS ONE** — vessel-collision mortality of blue/
    humpback/fin whales on the **U.S. West Coast** (open access; directly on-region — the key one).
  - **Conn & Silber 2013, Ecosphere** — vessel speed vs strike risk (open access).
  - *(optional)* **Laist et al. 2001** — "Collisions between ships and whales" (foundational
    review) or **Vanderlaan & Taggart 2007** — the speed→lethality curve.
- **Where:** PLOS ONE / Ecosphere (open) · Google Scholar for the others.
- **Format:** PDF · **Effort:** medium — **the only unbounded source; cap it at 2–3** ·
  **License:** open-access only (avoid paywalled copies).
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

- [ ] **SARs** — download the CA/OR/WA humpback SAR PDF(s). *(start here — backbone)*
- [ ] **Sanctuary** — Channel Islands Condition Report + Management Plan PDFs (+ others if in scope).
- [ ] **Whale Safe** — save the methodology page + the SB Channel / SF report cards.
- [ ] **Reviews** — Rockwood 2017 + Conn & Silber 2013 (open-access PDFs); stop at 3.
- [ ] **OBIS density** — decide: aggregate existing `sightings`, or pull OBIS API. *(separate tool track)*
- [ ] *(parked)* GEBCO — skip unless a demo needs the visual.

Drop the acquired files somewhere stable (e.g. `~/Source/DATA/oceans/corpus/<source>/`); the
B3 ingestion job will chunk + embed them into `doc_chunks`. Then F4 has a target it can hit.
