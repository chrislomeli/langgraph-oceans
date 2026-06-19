# `docs/design/` — the conservation-risk agent: data & tool design

The design specs for **what the agent answers, how it reasons, the tools, and how each data
collection is stored/chunked.** (Distinct from `docs/research/` = history / experiment logs /
the ML-track designs.) Live status is `docs/build-status.md`; product vision is
`docs/ocean-mammal-conservation-vision.md`. Most of this was designed 2026-06-19.

## Reading order (top-down)

| # | Doc | What it settles |
|---|---|---|
| 1 | **`scope-and-coverage.md`** | The **lane**: a conservation-risk reasoning agent, *not* general whale Q&A. Three levels of zoom (this whale / its kind / its waters); answers are *chains* across them; honest coverage + the in-scope test. **Read first.** |
| 2 | **`agent-orchestration-design.md`** | How we talk to the agent so it *chains*: the **hub-and-zoom mental model** + the locked 3-part tool-ad template (Trigger / Anti-overlap / Returns). Decision: small tools + recipe, not a scripted composite. |
| 3 | **`tool-design.md`** | The **6 Layer-3 tools** as Ad + **Query** + **Storage**. The query finalizes the storage → only **2 new tables** needed (`stock_status`, `doc_chunks`); the embedder dimension `D` is the last open decision. |

## The data collections (the corpus)

| Doc | Collection | Shape |
|---|---|---|
| `rag-collections.md` | the working list (index of all 6) | — |
| `rag-corpus-shopping-list.md` | what to acquire + demo questions | acquisition spec |
| `rag-preprocessing-design.md` | chunk + annotate strategy (cross-collection) | mechanics |
| `sar-collection-design.md` | **#1 SAR** | card (`stock_status`) + chunks |
| `sanctuary-condition-collection-design.md` | **#2 Condition Reports** | chunks only; selection-dominated |
| `sanctuary-mgmt-collection-design.md` | **#3 Management Plans** | chunks only; heterogeneous files |
| `reviews-collection-design.md` | **#4 Reviews** | chunks only; join on the *question* |

*(Collections #5 AIS and #6 `obis_seamap_points` are structured — designed in `tool-design.md`,
not as text collections.)*

## Status

All 4 text collections **designed, not built**. Remaining before build: decide **`D`** (the
embedder dimension) → build `stock_status` (hand-curate ~6 rows) + the `doc_chunks` ingestion.
