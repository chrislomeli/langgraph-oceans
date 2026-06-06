  The datasets — what to get, what to verify

  1. Photo-ID / image space — get this FIRST
  - Happywhale – Whale and Dolphin Identification (Kaggle, 2022). ~51k images, multi-species, labeled by individual_id.
  - Verify: (a) does the CSV carry date + location per image, or just image + individual_id + species? — my strong prior is no location/date in the Kaggle dump; confirm it. (b) the images-per-individual distribution — there's a long tail of individuals with a single image, which is your open-set reality and your hardest eval cases. (c) is there a usable train/test split.
  - Answers: enough to build + eval the photo_id tool in isolation and do the LoRA work — step 1 of the build, needs nothing else.
  
  2. Sighting histories — the hub's connective tissue (the risky one)
  - OBIS-SEAMAP (Duke) — megafauna occurrences, downloadable by taxon.
  - Verify: it's keyed to species + lat/lon + date, almost never to a named individual. So "this Happywhale individual's sightings" will not join here. Confirming this negative is the point.
  - Happywhale.com — the platform, not the Kaggle dump. The live platform tracks individuals with encounter histories (date+location). Verify whether there's API/data access — if so, the hub may be self-contained inside Happywhale even though the Kaggle file isn't. This is the most important thing to check.
  - Flukebook / Wildbook — individual-ID platforms; some catalogs expose encounter histories. Verify access per catalog.

  3. The single-species rich catalog — the strong alternative
  - North Atlantic Right Whale catalog (Anderson Cabot Center / New England Aquarium; ~340 individuals, each with a sighting history).
  - Verify: access terms — it's research-grade, likely a data request/agreement, not a one-click download. Also note NARW are ID'd by callosity/head patterns, not flukes, so the image modality differs.
  - Why it matters: it bundles individual + sightings + the ship-strike framing in one coherent source — making your AIS multi-hop the real conservation question, not a demo.

  4. Vessel traffic — AIS
  - Marine Cadastre AIS (marinecadastre.gov; NOAA/BOEM), free US AIS by year/zone.
  - Verify: coverage for your region/time, and that you can aggregate to a density grid (you'll never use raw points). Plan to subset by area+time — files are large.

  5. Knowledge corpus — text RAG
  - NOAA Marine Mammal Stock Assessment Reports (SARs) — public PDFs, per stock.
  - Verify: granularity — they're stock/population-level (abundance, mortality, PBR), not individual-level. Confirm → then "what's known about this individual" adapts to "…this stock," still fully grounded.

  What you're really looking for (the synthesis)

  One fork decides the shape:

  - If Happywhale-the-platform (or NARW) gives you individual → encounter history (date+loc) → the hub is per-individual: photo-ID → that whale's range → AIS. Cohesive, high-wow.
  - If only the Kaggle dump (identity, no loc/date) + OBIS (species-level, no individuals) are practically available → the hub drops to per-species/stock: photo-ID still gives identity, but range, "what's known," and the AIS overlap operate at the species level. Still a real, buildable, grounded project — just slightly less "this exact animal."

  Both are viable. The recon's only job is to tell you which, and that resolves your "real bar," species/region, and corpus-sourcing open items in one go.

  Priority order (per the components-first lesson)
  
  You don't need all five to start. Get them in build order:
  1. Happywhale (Kaggle) — unblocks the most AI-play (the photo_id component + LoRA), needs no hub. Start here.
  2. Resolve the hub question — check Happywhale-platform access, then OBIS, then NARW. This is the go/no-go.
  3. AIS + SARs — only needed once the agent's multi-hop and RAG come online (steps 2–3).
  
  So: download Happywhale today; spend the recon proving/refuting the hub join; defer AIS and SARs. Want me to drop this dataset checklist into the note's "Next step" section so it's there when you start?

---

## Recon results — RESOLVED 2026-06-04, then CORRECTED 2026-06-04

**Go/no-go on the per-individual hub join: GO — the open hub exists.**

> ⚠️ This section was first written as a **NO-GO** (see "How the first
> conclusion was wrong," below). Loading the actual OBIS-SEAMAP datasets into
> Postgres overturned it. The hub is real, open, and richer than hoped. The
> earlier NO-GO is preserved as a record of the reasoning, but it is **wrong** —
> read the correction first.

### The finding: Happywhale publishes its encounter data INTO OBIS-SEAMAP, openly

OBIS-SEAMAP hosts **Happywhale-contributed datasets** (`institution = Happywhale`,
`provider = "<person> via Happywhale.com"`) that carry the full Darwin Core
schema — including the per-individual fields that the generic species-occurrence
download lacks. Verified directly in the data:

| Field | Value | Meaning |
|---|---|---|
| `organism_id` | `https://happywhale.com/individual/88422` | **stable per-individual identifier** (the whale's HW profile) |
| `external_resource` | `https://au-hw-media-m.happywhale.com/<uuid>.jpg` | **the actual fluke photo** for that encounter (medium, ~117 KB) |
| `external_resource_thumb` | `…-t.happywhale.com/<uuid>.jpg` | thumbnail of same (~6 KB) |
| `latitude` / `longitude` / `date_time` | real | where/when of that sighting |

So a **single open dataset** gives the entire bridge in one namespace:
`organism_id (identity) → many (date, lat/lon) rows → + photo URL`.

**The numbers (humpback `dataset_1765` alone):** 206,685 sightings · 192,156 with
`organism_id` · **31,979 distinct individuals** · **20,550 seen >1×** (real
encounter histories) · up to **703 sightings on one animal** · ~11k seen once
(a natural open-set tail). The user's PNW-2020+ query returned 111,083 humpback
rows with `organism_id`.

**Verifications (all green):**
- ✅ `organism_id` is a real, repeating per-individual key with multi-sighting histories.
- ✅ photo URLs publicly fetchable (`HTTP 200`, `image/jpeg`; no auth, CORS-open).
- ✅ **license open and clean:** of identified rows, **CC0 ≈ 107,202** (public
  domain) plus ~6,900 other Creative Commons (BY / BY-SA / BY-NC / BY-NC-SA);
  exclude ~33 BY-NC-**ND**, ~2,201 blank, ~277 "no use without permission."
  Filter: `license LIKE '%creativecommons%' OR license LIKE '%publicdomain%'`
  (or CC0-only for zero strings attached). Attribution via `provider` /
  `rights_holder`.

This is **legitimate and reproducible** — published to a public biodiversity
repository under open licenses, `external_resource` is the Darwin-Core associated-
media link *meant* to be fetched. NOT scraping the gated platform.

### Decision (corrected)
1. **The real per-individual project is fully unblocked.** Identity + sightings +
   ranges + fetchable photos, all real, one namespace, openly licensed. **No
   synthetic linkage needed** (the earlier fabrication plan is dropped).
2. **The emergent agent is back on real data:** disambiguation across real
   look-alikes, multi-hop *individual* range → AIS, open-set recovery on the real
   single-sighting tail.
3. **Kaggle `train.csv` is now optional** — its anonymized hashes do **not** join
   to `organism_id` (different namespace, renamed images). Keep only as extra
   embedder-training images + the public leaderboard benchmark; the catalog is
   built from the OBIS-Happywhale images (`external_resource`), which carry real
   IDs *and* locations.

### How the first conclusion was wrong (kept as the lesson)
The initial recon checked the **front doors** and called it NO-GO:

| Source | Status | Still true? |
|---|---|---|
| Happywhale **platform** | gated (no API, view-only) | ✅ true — but irrelevant: the data flows *out* to OBIS openly |
| Flukebook / Wildbook | owned encounters, reciprocal collab | ✅ true |
| NPPID paper | routes back to Happywhale, MOA, 0% bulk | ✅ true |
| NARW catalog | data agreement, callosity-ID | ✅ true |
| OBIS-SEAMAP (generic **species** download) | `organism_id` empty | ✅ true *for that file* |

**The miss:** generalizing "`organism_id` ~empty for cetaceans" from the one
species-occurrence file, without discovering the **Happywhale-published datasets
within OBIS-SEAMAP** that populate it richly. The lock was on the platform's front
door; the data was sitting open in the public repository the whole time. Lesson:
load the actual datasets and look — provider/institution and the Darwin Core
extension columns vary per dataset.

### Do NOT scrape Happywhale (still stands)
The open path above makes scraping unnecessary *and* it remains the wrong move:
breaks reproducibility, the platform gates data on purpose, and it's a JS SPA over
a gated API. Use the OBIS-published data, which is the sanctioned, citable source.