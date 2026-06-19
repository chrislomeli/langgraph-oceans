# Sanctuary Condition Report design (collection #2)

*Status: designed (2026-06-19). The second collection. Grounded in the Channel Islands CR
(TOC + framework read). Mirrors `sar-collection-design.md` but the shape is different: **a huge
narrative doc where the dominant problem is selection, not chunking** — and **no structured
card.** Parents: `rag-collections.md`, `tool-design.md` (`doc_chunks` / `doc_search`),
`scope-and-coverage.md`. Join key = **`sanctuary`** (not `species`).*

## How it differs from SAR (the shape)

| | SAR (#1) | Condition Report (#2) |
|---|---|---|
| Size | ~6 pp/stock, mostly relevant | **482 pp**, ~90% **out of lane** |
| Structured card | yes (`stock_status`) | **no** |
| Dominant problem | chunk the prose | **select the 3% that's in-lane** |
| Join key | `species` | **`sanctuary`** |
| Value | facts + reasoning | **narrative only** (ship-strike *pressure* on the place) |

The CR is a glossy, figure-heavy report (~22 MB) covering water quality, kelp, habitat, Chumash
ecosystem services, archaeology, recreation, plus dozens of appendices. It has the same saving
grace as SAR — a **stable NOAA template (the DPSER framework) + a TOC with page numbers** — so
the lane content is named and locatable.

## Card? No *(LOCKED)*

CR value is *narrative* ("is this place under ship-strike pressure, how bad"), not citable
numbers. Its one quasi-structured datum — the vessel-traffic **pressure rating** (qualitative
good→poor + trend) — is fuzzy **and largely redundant** with the `vessel_traffic` tool, which
already gives the *actual* AIS transit numbers. So **CR = chunks only.** `sanctuary` is the join
key (known upstream from `sighting_context` / `vessel_traffic`), not a stored fact. *(If a demo
ever wants "the sanctuary rates vessel-traffic pressure as X," lift that one rating then.)*

## Chunks = aggressive selection

**Keep only the in-lane sections; drop everything else.** Keep-set:

| Section (CINMS page refs) | Why |
|---|---|
| Driving Forces & Pressures → **Vessel Traffic** (p47) | the *pressure* — shipping crossing the sanctuary, ship-strike framing. **The core.** |
| Driving Forces & Pressures → **Ocean Noise** (p52) | whale-relevant pressure |
| State of Drivers & Pressures → **Pressure Ratings** (p66) | the severity rating (good→poor + trend) for those pressures |

**Drop:** water quality, habitat/kelp, living resources, Chumash ecosystem services, maritime
archaeology, recreation, ecosystem-services, all appendices, the DPSER framework boilerplate —
~90% of the document. *(Living Resources / place-specific cetacean status deliberately left out
of v1: the SAR already gives species status, and the CR's unique value is the place-pressure
framing. Easy to add later.)*

**Annotation per chunk:** join key **`sanctuary`** (one of the 6 NMS names), `doc_type =
'condition-report'`, `section`, `source`, `year`; `species` left broad/null (a sanctuary hosts
many species — it's a place, not a stock). **Contextual header:** `[Channel Islands NMS
Condition Report — Vessel Traffic] <text>`.

## The seam — CR vs. Management Plan *(LOCKED)*

The CR *also* has a "Response → Vessel Traffic" section (p209) describing mitigation — which is
the **Management Plan's** job (#3). To keep `doc_type` roles clean and avoid near-duplicate
chunks:

> **CR owns the *pressure / severity* side; the Mgmt Plan owns the *response / mitigation* side.**

So we **drop the CR's Response section** and let #3 carry "what's being done." Matches the
corpus README's intended split (CR = severity context, Mgmt = mitigation branch).

## Mechanics — the TOC rescues us

Don't extract 482 pages. With only **5 condition reports**, the cheapest, most bulletproof move:

1. **Hand-map the in-lane page ranges per doc** (5 docs → trivial; e.g. CINMS Vessel Traffic
   ≈ pp 47–51, Ocean Noise ≈ pp 52–53, Pressure Ratings ≈ pp 66–74). The TOC gives these
   directly.
2. **Extract + clean only those ~15 pages** — sidesteps the whole-document figure/caption
   nightmare. Cleaning: strip running headers/footers + page numbers, drop figure captions
   that are pure furniture (keep ones that carry a vessel/whale fact), de-hyphenate, collapse
   whitespace.
3. **Recursive-split within** (~300–500 tok, ~50 overlap), never crossing a section boundary.
4. **Eyeball all 5** outputs.

Same "tiny corpus → bespoke beats generic" lesson as SAR — here it's **page-range selection**
instead of heading-regex, because the doc is huge but the corpus is tiny and the TOC is a
structured index.

## The payoff

The CR supplies the agent's *place-level severity framing*: "the Channel Islands sanctuary
rates vessel traffic as a significant and increasing pressure, in the SB Channel where the
shipping lanes run." Crossed with `vessel_traffic`'s actual AIS numbers (the precise overlap)
and the SAR's species strike-vulnerability, it's the **"is the risk real *here*"** layer of the
F5 chain — with the **Mgmt Plan (#3)** next supplying "and is anything being done about it."
