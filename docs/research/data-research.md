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

✻ Churned for 1m 4s