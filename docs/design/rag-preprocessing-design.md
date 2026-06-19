# Design note — RAG corpus preprocessing (chunk + annotate)

*Status: design, not built. Precursor to the `doc_chunks` ingestion (build-status **B3**).
Mid-level — strategy, the metadata schema, and build order. Not code. Companion to
`rag-corpus-shopping-list.md` (the acquired corpus) and the agent design in
`ocean-mammal-conservation-vision.md` (Layer 3 `hybrid_search`, Layer 4 `doc_chunks`).*

## Why this exists (the load-bearing realization)

By the time the agent queries the corpus, upstream tools have **already** told it the
**species** (from `photo_id`) and the **sanctuary** (from `vessel_traffic.overlapping_zones`).
So its queries are *scoped*: "humpback + mortality threats," "Monterey Bay NMS + what's being
done about strikes" — not "something about whales."

The failure mode this guards against: **all nine SARs read almost identically** (same sections,
vocabulary, prose). A bare vector search for "population trend and ship-strike mortality" on a
humpback question will return **blue whale** chunks — textually near-identical. Without
metadata the agent can't scope to the right species, and RAG silently mixes stocks.

**Conclusion: the annotation matters more than the chunking strategy.** Chunks must be
retrievable *by the join keys the upstream tools already produced* (species, sanctuary), or the
corpus is unusable no matter how cleanly it's split. "Indiscriminate" is out; "by heading" is
the right base but only half the job.

## Chunking strategy

- **Section/heading-aligned.** The SARs have a stable schema (Stock Definition · Population
  Size · Min Population · Trend · PBR · **Human-Caused Mortality** · Fishery Info · Status) that
  maps directly onto the agent's topic-scoped queries. Chunk on those boundaries.
- **Target ~300–500 tokens, ~50 overlap**, but **never cross a section boundary** and **never
  split a table mid-row**. Tiny sibling sections under one heading may merge; an oversized
  section splits within itself (keeping the heading on each piece).
- **Contextual header prepended to the embedded text** (cheap, high-value): embed as
  `[Humpback CA/OR/WA SAR — Human-Caused Mortality] <text>` rather than bare text, so even pure
  vector search carries the species+topic signal and citations are precise. Stored in `header`.

## Metadata schema (per chunk) — the actual lever

The **join keys** (★) tie a chunk back to what the upstream tools produced:

| Field | Example | Why |
|---|---|---|
| ★ `species` / `common_name` | `Humpback Whale` | joins to the `photo_id` individual's species |
| ★ `sanctuary` | `Monterey Bay NMS` | joins to `vessel_traffic.overlapping_zones` |
| `doc_type` | `SAR` / `condition-report` / `mgmt-plan` / `review` | scope by *kind* of question |
| `stock` / `region` | `California/Oregon/Washington` | the right stock, not a sibling |
| `section` | `Human-Caused Mortality` | topic scoping |
| `source`, `year` | `Humpback CA/OR/WA SAR`, `2021` | citation + recency |

Most of these are **derivable for free** from the curated filename (e.g.
`humpback-caorwa-2021.pdf` → species + stock + year + doc_type) plus the section heading — so
tagging is mechanical, not a labeling project.

## Maps onto the existing `doc_chunks` schema (not new infra)

The vision's `doc_chunks` already has the slots: `species text[]`, `region`, `source`,
`embedder_ver`, **and a `tsv` for keyword/hybrid search**. Populating those columns *is* this
job — indiscriminate chunking would waste a schema designed for metadata-scoped hybrid
retrieval. Add `doc_type` / `sanctuary` / `section` / `header` columns alongside.

**Retrieval = hybrid:** metadata pre-filter (species/sanctuary/doc_type from the agent's known
context) → vector similarity **+** `tsv` keyword rank → top-k. This is exactly the `Filters`
contract (species, region, source) applied to text.

## Per-doc-type handling

- **SARs** (9) — cleanest. Stable sections → section chunks; species/stock/year from filename.
  The combined Pacific SAR must be **split per-stock first** (each stock is a chapter); prefer
  the per-species files for clean attribution, use the combined doc only for coverage gaps.
- **Sanctuary** Condition Reports + Mgmt Plans (10) — narrative, figure-heavy. Heading chunks
  still work; key annotation is `sanctuary` (the join key) + `doc_type` (condition vs mgmt).
- **Reviews** (2) — papers. Chunk Abstract / Results / Discussion; **skip Methods & References**.
  Lighter scoping (mechanism is species-agnostic); tag `topic` (mechanism / mortality), region.

## Two gotchas (decide deliberately, don't discover)

1. **Tables are the highest-value, hardest content.** The SAR vessel-strike-vs-entanglement
   *numbers* live in tables that `pdftotext` mangles. **Decision:** rely on the prose sentence
   that states the section total (SARs restate key figures in text, e.g. "…total 54.75 whales
   for 2015–2019"), and linearize *only* the abundance/mortality summary tables. Don't try to
   parse every fishery-by-fishery table — that granularity isn't what the agent asks for.
2. **Condition reports are figure-heavy (20–37 MB)** → text extraction yields page furniture
   (headers, captions). A light cleaning pass: strip repeated headers/footers, de-hyphenate
   line-break splits, collapse whitespace. Keep figure captions (they sometimes carry the fact).

## Build order

1. This design note. →
2. **Extractor + section segmenter** per doc_type → clean text + `(section, text)` spans.
   Smoke on 1 SAR + 1 condition report + 1 review.
3. **Chunker** — section-aware, token budget, contextual headers.
4. **Metadata tagger** — species/stock/year/doc_type from filename; sanctuary/section from
   content. Mostly mechanical.
5. **Embed + load** → `doc_chunks` (text embedder + `tsv`). This is the B3 ingestion.
6. **Retrieval smoke (the checkpoint):** a scoped query ("humpback mortality") returns
   **humpback-only** chunks from the Mortality section — proving the metadata scoping works,
   not just that vectors load.

## The honest scope

21 docs, consistent structure → bounded, not a sprawl. The discipline that earns its keep is
the **metadata tagging + contextual headers**; the chunking itself is ordinary. Skip either and
the corpus quietly mixes stocks — which the agent would then cite confidently and wrongly.
