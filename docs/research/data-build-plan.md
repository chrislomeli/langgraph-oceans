# Data Build Plan — standing up the whole database

> Goal: get **the whole database populated** (real identities + sightings +
> ranges + traffic) before any ML/agent work. You should be able to open the DB
> and see every table with rows in it. Embeddings and the text corpus (the
> ML/RAG content) come *after* this.

Status: planning. Created 2026-06-04. Rewritten 2026-06-04 (real hub), and again
2026-06-04 (**lossless storage** — see the principle below).

> **What changed (real hub):** OBIS-SEAMAP's Happywhale-contributed datasets carry
> `organism_id` (the real individual), `external_resource` (the real fluke photo),
> and real lat/lon/date. Nothing is fabricated; everything is real, openly licensed
> (CC0-dominated), citable. See `data-research.md` → "Recon results (CORRECTED)".

---

## 0. Storage principle (load-bearing) — ingest broad & lossless, narrow downstream

We will **not throw away data at ingest for size reasons.** Every cut we keep has
a *domain* reason (different population, outside AIS coverage, restrictive license);
**size is never a reason to discard.** Postgres + PostGIS hold the full data fine.

- **Region** and **lane-threshold** and **AIS year** are **reversible query-time
  choices** (views / `WHERE` / `ST_Clip`), *not* ingest filters. We've reshaped this
  project many times in one day — so we keep all options open and let the **real
  agent query patterns** (once we build the tools) decide what to materialize.
- AIS is stored as a **PostGIS raster** (lossless, compact). "Lanes" is a
  rebuildable materialized view; "region" is an `ST_Clip`. Nothing is deleted.

---

## 1. Decisions

| Decision | Status | Value / Why |
|---|---|---|
| **Species** | locked | **Humpback** (*Megaptera novaeangliae*) — densest individual data, ship-strike poster species |
| **Source of hub** | locked | **OBIS-SEAMAP Happywhale datasets** — real `organism_id` + sightings + photo URLs, one namespace |
| **License filter** | locked (domain) | `license LIKE '%creativecommons%' OR '%publicdomain%'` — drop BY-NC-ND / blank / "no use without permission" |
| **Population scope** | locked (domain) | **Eastern North Pacific** — drop western Pacific (China/Okhotsk/Philippine = different population) |
| **Coordinate system** | locked | Catalog in EPSG:4326 (geography); AIS raster kept native **EPSG:3857**, transform at query |
| **Demo region** | **DEFERRED — reversible** | candidates: CA 32–39 (shelf-bounded: SoCal+Monterey+SF) or broader West Coast. A query filter, **not** an ingest gate. Driven by *whales*, not AIS size |
| **AIS year** | **DEFERRED — reversible** | store all downloaded (2022–2025); pick representative (2023) at query time |
| **Lane threshold** | **DEFERRED — reversible** | a view (`vessel_count >= N`), redefinable; full raster retained |

The data shelves (for when we *do* pick a demo region): clear humpback density
peaks at **SoCal/SB (33–34)**, **Monterey (36, richest)**, **SF (37–38)**,
**WA (48)**, separated by genuine sparse zones at **24–31** and **39–47**. AIS
covers the US coast to **~51.7°N** (excludes SE Alaska). These inform the *demo*
region; they do not gate ingest.

---

## 2. The target database

`datasets.*` schema (one Postgres, PostGIS + pgvector + **postgis_raster**).

| Table | Holds | Source | Phase |
|---|---|---|---|
| `individuals` | named whales (HW id, species) | OBIS-Happywhale `organism_id` | **DB build** |
| `sightings` | encounters: individual + where/when + photo URL + license | OBIS-Happywhale rows | **DB build** |
| `ais_2022…ais_2025` + `ais_all` view | **full** transit-count grid per year (lossless raster), unified by a `year`-column view | Marine Cadastre Transit Counts | ✅ **loaded** (`datasets`) |
| `vsr_zones` | voluntary 10-kt speed-zone polygons | NOAA sanctuaries GIS (Whale Safe layer) | **DB build** |
| `ais_lanes` *(matview)* | derived "busy cells" for the demo region | view over `ais_all` | reversible, optional |
| `fluke_embeddings` | image vectors keyed to individual | OBIS-Happywhale photos | *later — ML phase* |
| `doc_chunks` | text corpus chunks + vectors | NOAA SARs, Whale Safe, policy | *later — RAG phase* |

### DDL
```sql
-- everything lives in the existing `datasets` schema (holds the OBIS source + AIS)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;

CREATE TABLE datasets.individuals (
  individual_id text PRIMARY KEY,   -- the Happywhale individual URL (organism_id)
  species       text,
  catalog       text DEFAULT 'happywhale-via-obis',
  n_sightings   int,
  first_seen    date,
  last_seen     date
);

CREATE TABLE datasets.sightings (
  sighting_id   bigserial PRIMARY KEY,
  individual_id text REFERENCES datasets.individuals(individual_id),
  species       text,
  obs_date      date,
  location      geography(Point,4326),
  group_size    int,
  image_url     text,               -- external_resource (the fluke photo)
  thumb_url     text,               -- external_resource_thumb
  license       text, rights_holder text, provider text,
  source        text DEFAULT 'OBIS-SEAMAP (Happywhale)',
  source_row_id text
);
CREATE INDEX ON datasets.sightings USING gist(location);
CREATE INDEX ON datasets.sightings (individual_id);

-- AIS: LOSSLESS tiled raster, native 3857. Loaded as one table per year by
-- raster2pgsql (creates rid serial PK, rast raster; GiST footprint index via -I),
-- then unified by a year-column view. No clip, no threshold. Lives in `datasets`.
--   raster2pgsql -s 3857 -t 256x256 -I -C -Y ais-transit-count-2024.tif datasets.ais_2024 | psql ...
CREATE VIEW datasets.ais_all AS
  SELECT 2022 AS year, rast FROM datasets.ais_2022
  UNION ALL SELECT 2023, rast FROM datasets.ais_2023
  UNION ALL SELECT 2024, rast FROM datasets.ais_2024
  UNION ALL SELECT 2025, rast FROM datasets.ais_2025;
-- ⚠ all years share the same grid → every spatial query MUST filter `WHERE year = …`
--   (otherwise overlapping footprints from multiple years multi-count).

CREATE TABLE datasets.vsr_zones (
  zone_id text, name text, season text,
  geom    geography(Polygon,4326)
);
CREATE INDEX ON datasets.vsr_zones USING gist(geom);
```

---

## 3. What we have vs. what we still need

**Have:**
- [x] **OBIS-SEAMAP Happywhale datasets in Postgres** (`obis_seamap_points`) — the hub.
- [x] **AIS Transit Counts 2022, 2023, 2024, 2025** `.tif` in `~/Downloads/AIS/` (verified: GTiff UInt32, 100 m, correct product).
- [x] Kaggle `train.csv` + images — **optional** (embedder training / leaderboard; does NOT join the hub).

**Still need:**
- [ ] **`postgis_raster` extension** enabled (confirm).
- [x] **VSR-zone geometry — availability checked (2026-06-04):** no dedicated VSR
      shapefile exists, but **NMS sanctuary boundaries are downloadable** (ESRI Shape
      + KML) at [sanctuaries.noaa.gov GIS](https://sanctuaries.noaa.gov/library/imast_gis.html)
      and serve as a faithful proxy (the VSR program operates within them). Exact zone
      corners are transcribable from the annual NOAA notice / Blue Whales Blue Skies if
      wanted. **Non-blocking** (vsr_zones is a complement, not the foundation).
- [ ] **NOAA humpback Stock Assessment Report** (PDF) — *RAG phase*.
- [ ] **Whale Safe methodology + report cards** — reference + *RAG phase*.
- [ ] **Photo download** of `external_resource` — *catalog build, scoped to demo region when chosen*.

*(No "AIS year" or "region lock" needed up front — both are deferred, reversible.)*

---

## 4. The build steps

### Step 0 — schema
Run the DDL above (incl. `postgis_raster`).

### Step 1 — OBIS-Happywhale → `sightings` + `individuals`  ✅ DONE
Built `sightings` from the raw `obis_seamap_points` (one row per encounter, open
license, real `organism_id`), then derived `individuals` from `sightings`, then
wired the FK. Kept **all species** (species is a column, filter at query time); no
region/date ingest filter (reversible `WHERE` clauses). Reused the `location`
geometry already added to `obis_seamap_points`.
```sql
-- sightings: clean encounters, open-licensed only
INSERT INTO datasets.sightings
  (individual_id, species, obs_date, location, group_size,
   image_url, thumb_url, license, rights_holder, provider, source_row_id)
SELECT organism_id, scientific_name, NULLIF(date_time,'')::date, location, group_size,
       external_resource, external_resource_thumb, license, rights_holder, provider, row_id
FROM datasets.obis_seamap_points
WHERE organism_id IS NOT NULL
  AND license LIKE ANY (ARRAY['%creativecommons%','%publicdomain%']);

-- individuals: one row per whale, derived from sightings
INSERT INTO datasets.individuals (individual_id, species, n_sightings, first_seen, last_seen)
SELECT individual_id, mode() WITHIN GROUP (ORDER BY species),
       count(*), min(obs_date), max(obs_date)
FROM datasets.sightings GROUP BY individual_id;

-- wire the relationship
ALTER TABLE datasets.sightings
  ADD CONSTRAINT sightings_individual_fk
  FOREIGN KEY (individual_id) REFERENCES datasets.individuals(individual_id);
```
- ✅ **114,126 sightings** across **23,738 individuals** — avg 4.8 sightings; **9,624 seen once** (the open-set tail); most-resighted **505**. The open-license filter dropped 2,201 unusable rows (null license = the same photo-less rows = 143 individuals with no usable record).

### Step 2 — AIS → per-year tables + `ais_all` view  ✅ DONE
Loaded each annual GeoTIFF **whole** as its own tiled raster table (lossless), then
unified with the view. Note: `raster2pgsql` **skips all-NODATA tiles**, so the row
count = tiles that actually carry traffic (open ocean isn't stored — lossless for data).
```bash
for y in 2022 2023 2024 2025; do
  raster2pgsql -s 3857 -t 256x256 -I -C -Y \
    ~/Downloads/AIS/ais-transit-count-$y.tif datasets.ais_$y \
  | psql "postgresql://localhost:5432/oceans"
done
# then: CREATE VIEW datasets.ais_all (see DDL above)
```
- ✅ Loaded: `ais_2022` 24,637 · `ais_2023` 27,302 · `ais_2024` 25,917 · `ais_2025` 26,863 tiles; all native 3857, 100 m, 256×256; full national extent (West Coast covered).
- ✅ **Range→AIS multi-hop verified** on real data (whale `individual/479`: range over 6 tiles, peak 9,290 transits/yr in its busiest cell).

### Step 3 — protected zones → `vsr_zones`  ✅ DONE
No dedicated VSR shapefile exists, so loaded the **NMS sanctuary boundaries** as the
protected-area proxy (the VSR program operates within them), from
[sanctuaries.noaa.gov GIS](https://sanctuaries.noaa.gov/library/imast_gis.html).
Loaded 7 sanctuaries via `ogr2ogr` (reproject NAD83→4326, `PROMOTE_TO_MULTI`, name
**assigned per file** via `-sql "SELECT '<name>' …"` since field schemas vary and
GFNMS has no name field):
- Cordell Bank · Greater Farallones · Monterey Bay · **Chumash Heritage** · Channel
  Islands (SoCal) · Olympic Coast (WA) · **Hawaiian Islands Humpback** (breeding).
- ✅ 7 sanctuaries / **12 polygons**; areas validate vs official sizes (MBNMS 15,794 km²).
- Note: Hawaii is a sanctuary but lacks CA's 10-knot VSR — treat `vsr_zones` as
  "protected/management areas" broadly, not strictly speed zones.
- *(stored as `geometry(MultiPolygon,4326)`; intersect with whale ranges via `::geography` cast.)*

### Step 4 — photo catalog: `external_resource` images
Download per-sighting `image_url` (or `thumb_url` for bulk) to a blob dir; DB keeps
the URL. Scope the *download* to the demo region once chosen (bandwidth, not a data
cut — the URLs remain in `sightings` regardless). Embedding = ML phase.

### → The database is now "in place" ✅ — and queried losslessly
> **Three-layer multi-hop verified** end-to-end on real data (whale `individual/479`,
> 2024): range × AIS traffic × protected zones → peak ~31,471 transits, overlaps
> Channel Islands / Chumash / Monterey Bay NMS.
>
> **Two tool-phase refinements this surfaced** (query-side, data is correct):
> 1. `ST_Extent` (bounding box) overstates a wide-ranging whale's range — an outlier
>    sighting balloons the box over open ocean. Use a **convex hull / buffered points /
>    recent-only** for a truthful range when wrapping into the tool.
> 2. Dedupe zone names with `string_agg(DISTINCT …)` (CINMS is 2 polygons → listed twice).
```sql
-- a humpback's range (any region/time, computed on demand)
SELECT ST_Extent(location::geometry) FROM datasets.sightings WHERE individual_id = :id;

-- range × traffic, straight from the full raster (no pre-threshold needed):
SELECT (ss).max AS peak_transits, (ss).mean AS mean_transits, (ss).count AS busy_cells
FROM (
  SELECT ST_SummaryStats(ST_Clip(rast, ST_Transform(:range::geometry,3857))) AS ss
  FROM datasets.ais_all
  WHERE year = 2023
    AND ST_Intersects(rast, ST_Transform(:range::geometry,3857))
) q;

-- range × speed zone:
SELECT bool_or(ST_Intersects(:range::geography, geom)) FROM datasets.vsr_zones;

-- open-set tail:
SELECT count(*) FROM datasets.individuals WHERE n_sightings = 1;
```

### Optional — `ais_lanes` materialized view (reversible convenience)
Only if/when a tool wants pre-extracted "busy cells" for a demo region. Rebuildable
at any threshold/region; never the source of truth.
```sql
CREATE MATERIALIZED VIEW datasets.ais_lanes AS
SELECT r.year,
       (pp).geom::geography AS cell,
       (pp).val::int        AS vessel_count
FROM datasets.ais_all r,
     LATERAL ST_PixelAsPolygons(
       ST_Clip(r.rast, ST_Transform(:demo_region::geometry, 3857))
     ) pp
WHERE r.year = :year AND (pp).val >= 100;   -- year filter required (overlapping grids); threshold = a knob
```

### Step 5 — *later (ML/RAG phase)*
Embed staged fluke images → `fluke_embeddings`; ingest NOAA SARs + Whale Safe +
policy → `doc_chunks`.

---

## 5. Where Whale Safe sits (kept in the mix)

Not a data feed (dashboard/report-card only). Contributes: **framing** (risk =
presence × traffic × speed/zone), the **`vsr_zones`** layer (NOAA GIS, Step 3), and
**corpus material** (RAG phase). Real-world reference, never claimed to be
reproduced. AIS (Marine Cadastre) remains the required traffic source.

---

## 6. Definition of done (this doc)

- [x] Schema created (`postgis` + `postgis_raster` + `vector` enabled in `oceans`)
- [x] **`sightings` + `individuals` loaded** — 114,126 sightings / 23,738 individuals (all species, open license); FK wired
- [x] **AIS loaded** — `datasets.ais_2022/2023/2024/2025` (24.6K/27.3K/25.9K/26.9K tiles) + `ais_all` view; **lossless**
- [x] **`vsr_zones` loaded** — 7 NMS sanctuaries / 12 polygons (incl. Chumash, Olympic Coast, Hawaii)
- [x] **lossless queries verified** — three-layer multi-hop (range × AIS × zones) runs end-to-end on real data
- [ ] *(deferred, reversible)* demo region + AIS year + lane threshold — chosen later from real query patterns; photos downloaded for that region

**DATABASE IS IN PLACE** (2026-06-05) — real, openly-licensed, per-individual,
lossless. Remaining: the *later* ML/RAG phases (`fluke_embeddings`, `doc_chunks`)
and wrapping these proven queries into the agent tools. Region/threshold/year still
fully open.
