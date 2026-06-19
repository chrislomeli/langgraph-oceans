# SAR collection design (collection #1)

*Status: designed (2026-06-19). The first collection walked through the full design. SAR splits
across the two new storage homes — the **`stock_status` card** (facts) and the
**`doc_type=SAR` chunks** (narrative). Grounded in the humpback CA/OR/WA SAR (read in full).
Parents: `rag-collections.md` (the list), `tool-design.md` (the two tables), `scope-and-coverage.md`
(the lane). The keep/narrative split is the two-tier compression from `scope-and-coverage`.*

## The split in one line

**The card absorbs the *facts*; the chunks keep only the *reasoning the card can't hold*.** A
SAR is ~80% boilerplate; the species signal is a handful of numbers + a few reasoning
paragraphs. The card takes the numbers, the chunks take the paragraphs, everything else is
dropped.

---

## Part A — `stock_status` (the facts card)

**Key decision *(LOCKED)*:** key on **`species`**, default to the **primary ENP / CA-OR-WA
stock**. Two species map to two SARs each (humpback: CA/OR/WA *and* CASMEX DPS; killer whale:
ENP *and* Southern Resident), so the strict key is *species × stock*, and "which stock" depends
on the individual's range (a life-spoke dependency). For v1 we collapse to species/primary-stock
because the sightings DB is curated to those stocks, most individuals belong to them, and it
keeps the card a clean lookup on what the anchor provides. **The DPS/multi-stock complexity is
not lost — it lives in the narrative chunks** (the "Status of Stock" section explains the
CA/OR/WA feeding stock *mixes* endangered Central-America + threatened Mexico + unlisted Hawaii
DPSs). Stock-resolution can be added later if a demo needs it.

**Fields** (★ = agentic — drives a branch in the trace; humpback values shown):

| Field | Humpback CA/OR/WA | Role |
|---|---|---|
| `species` (PK), `stock` | Humpback Whale, California/Oregon/Washington | key + provenance |
| `abundance`, `abundance_cv`, `abundance_year`, `method` | 4,973 · 0.048 · 2015–18 · mark-recapture | "how many" + honesty |
| `n_min` | 4,776 | feeds the PBR logic |
| `pbr`, ★`pbr_us_waters` | 58.7 · **29.4** | jurisdictional subtlety (stock spends ~½ its time outside the EEZ) |
| ★`trend` | +8.2%/yr (increasing) | "recovering?" |
| ★`dominant_threat` | **entanglement** | **the F5 severity pivot** (humpback = entanglement-dominant; blue/fin = vessel strike) |
| ★`ship_strike_yr_observed`, ★`ship_strike_yr_estimated` | **1.76 · 22** | the honest detection gap |
| `entanglement_yr` | 24.9 | the other half of the mortality split |
| `total_human_mortality_yr`, ★`exceeds_pbr` | 48.3 · **true** | the "strategic stock / not sustainable" conclusion (48.3 > 29.4) |
| `esa_status`, `mmpa_status` | mixed DPS · strategic | "is it endangered?" |
| `source`, `year` | Humpback CA/OR/WA SAR · 2021 | citation |

**The starred fields are the point** — they let the agent pivot the investigation (entanglement
vs. strike), state the honest strike toll (1.76 observed → ~22 real), and conclude "exceeds the
sustainable limit." Everything else is supporting fact.

**The card holds numbers + the observed/estimated *pair*; the *explanation* of the gap stays in
the chunks.** (Card says "1.76 observed, 22 estimated"; chunk says "because detection is ≤10%,
via the Rockwood encounter model.")

**Build:** hand-curate ~6 rows (one per species, primary stock) from the 9 SAR PDFs. Small,
verifiable, one-time.

---

## Part B — `doc_type=SAR` chunks (the narrative)

Section-aligned chunks on the SAR's stable headings; since the card now holds the facts, keep
**only the sections carrying reasoning the card can't hold:**

| SAR section | Keep? | Why |
|---|---|---|
| **Vessel Strikes** | ✅ keep — **richest** | the *why* behind the gap (Rockwood model, ≤10% detection), 82%-of-mortality-in-10%-of-area, southern-CA winter/spring, speed/ECA effects — pure F5 fuel |
| **Status of Stock** | ✅ keep | DPS/listing complexity + the strategic-stock reasoning |
| **Human-Caused Mortality (intro) + Habitat Concerns** | ✅ keep | strike-vs-entanglement framing; noise / marine-heatwave threats |
| **Stock Definition / Geographic Range** | ✅ keep | migration/connectivity, movement probabilities — "where this population goes" |
| Population Size / Current Population Trend | ⚠️ keep **only the recovery-story prose** (~1,200 in 1966 → 18–20k by 2004–06) | a narrative the card's single "+8.2%/yr" can't convey; **drop the methodology/CV prose** |
| Net Productivity / PBR | ❌ drop | pure methodology boilerplate; numbers are in the card |
| Fishery Information **tables** | ❌ drop the grid | per the table decision; totals are in the card (keep nothing — the prose total is redundant with the card) |
| References | ❌ drop | — |

**Annotation per chunk** (the join keys + citation): `species` (one of the 6 common_names),
`stock`, `doc_type='SAR'`, `section` (the heading), `source`, `year`. **Contextual header**
prepended to the embedded text: `[Humpback CA/OR/WA SAR — Vessel Strikes] <text>` — so even pure
vector search carries the species+topic signal and citations are precise.

**Chunk sizing:** ~300–500 tokens, ~50 overlap, **never cross a section boundary**; SAR sections
are often short enough that one section = one chunk.

**Source-file rules:**
1. Ingest the **per-species SAR files** (already per-stock → clean citation provenance), **not**
   the combined 2024 Pacific doc (avoids near-duplicate chunks / muddied provenance).
2. **Two-SAR species** (humpback, killer whale): chunk **both** files, each tagged with its
   `stock`, so `doc_search` can surface the CASMEX / Southern-Resident narrative even though the
   card picked the primary stock.

---

## Mechanics — how we actually chunk

**Not a generic chunker, not an LLM — a small bespoke segmenter, because the corpus is tiny
(9 docs) and regular (one NOAA template).** A generic "heading-aware" splitter fails here: the
SAR is a PDF, so extraction yields a flat character stream with no heading markers — "VESSEL
STRIKES" arrives as an ordinary line amid page numbers and two-column reflow. But we already
**know** the ~10 section headings, so the job flips from *detect unknown structure* (hard) to
*find these known strings* (easy). Three layers:

1. **Extract + clean — the actual hard part.** `pdfplumber` / `pdftotext` → cleaning pass: strip
   repeated headers/footers + page numbers, de-hyphenate line-break splits, collapse whitespace,
   handle the page-1 two-column reflow. Most pain lives here, not in the splitting.
2. **Split on the known heading list — the meaningful boundaries (a cheap rule, ~15 lines of
   regex).** Case-insensitive match on the expected SAR headings. NOAA helps: **major sections
   are ALL-CAPS, subsections Title-Case** → casing is a second cue (e.g. "Vessel Strikes" is a
   subsection under "HUMAN-CAUSED MORTALITY").
3. **Sub-split over-long sections with a standard recursive character splitter** (~300–500 tok,
   ~50 overlap) — **never crossing a section boundary.** The utility does the boring work; the
   bespoke rule owns the boundaries.

**No LLM / semantic chunking:** it's a 9-doc, one-time job — deterministic + **eyeball all 9
outputs** beats a non-deterministic pass you'd have to re-verify. (Corpus size + regularity
change the answer: the general RAG literature obsesses over chunking because it faces millions
of arbitrary docs; we have nine with a template.)

**Design ↔ mechanics alignment:** our keep/drop list keeps the **prose** sections and drops the
**tables** — and tables are exactly the content that's miserable to chunk. So the keep/drop
decision already dodges the hardest chunking; we only ever split clean prose.

---

## The payoff

The starred card fields + the **Vessel Strikes** chunk are what make the F5 ship-strike story
sing: the card gives the agent "entanglement-dominant, but strikes ~22/yr and the stock already
exceeds its sustainable limit," and the chunk gives the *why and where* ("≤10% detected; 82% of
strike deaths in 10% of the range; worst in southern CA in winter/spring") — which the agent
then crosses with the individual's actual range + traffic. Facts from the card, reasoning from
the chunk, risk from the intersection.
