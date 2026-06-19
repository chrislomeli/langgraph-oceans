# Ship-strike Reviews design (collection #4)

*Status: designed (2026-06-19). The last text collection, and the simplest. Two open-access
papers (capped — the "garnish / weakest-agency" mechanism source). Grounded in both abstracts.
Different from #1–#3 in one way: **the join axis is the *question/mechanism*, not species or
sanctuary.** Parents: `rag-collections.md`, `tool-design.md`, `scope-and-coverage.md`.*

## The two papers (abstracts carry the value)

| Paper | Headline findings | Role |
|---|---|---|
| **Rockwood 2017** (PLOS ONE; West Coast, blue/humpback/fin) | strike mortality **7.8× / 2.0× / 2.7×** the recommended limit → impediment to recovery; **74/82/65% of mortality in 10% of the area**; risk highest in SF & Long Beach lanes | magnitude + spatial concentration. **Literally the paper the SAR cites** for its "~22 estimated strikes / 82%-in-10%" numbers → the primary evidence under the `stock_status` card. |
| **Conn & Silber 2013** (Ecosphere; N. Atlantic right whale) | vessel speed restrictions cut strike mortality **80–90%**; strong speed→lethal-injury relationship; "speed limits are a powerful tool" | the **"does slowing ships help?"** mechanism. ⚠️ right-whale/East-Coast study, but a *mechanism* paper that generalizes to large whales — record the caveat; the agent should cite it as cross-species evidence. |

## Card? No

Mechanism/narrative papers, no per-stock facts (consistent with #2/#3).

## The different property — join on the *question*, not species/sanctuary

SAR keys on `species`, CR/Mgmt on `sanctuary`; **reviews key on the mechanistic question** —
*"why are whales strike-prone," "does slowing ships help"* — which the agent asks regardless of
species or place. So the menu is essentially **`doc_type='review'` + the semantic query**; with
only 2 docs, that's enough (no topic taxonomy needed). Light species tag where the paper is
species-specific.

## Chunks — keep it tight

**Keep Abstract + Discussion. Drop Methods + References.** (No Results — number/figure-heavy and
restated in the Abstract; no Intro — lit-review citations.) The Abstract alone is ~80% of the
value: dense, self-contained, all the headline findings. Discussion adds the recommendations
(lane modifications, speed reductions, Areas to be Avoided).

**Annotation:** `doc_type='review'`; `species[] = {Blue, Humpback, Fin}` for Rockwood, **null**
for Conn & Silber (agnostic); `section` (Abstract / Discussion); `source`; `year`;
`sanctuary` null. **Contextual header:** `[Rockwood 2017 — West Coast vessel strikes — Abstract]
<text>`.

## Mechanics — trivial

Two 2-column academic PDFs. Extract → locate "Abstract" / "Discussion" by heading → clean the
2-column reflow → chunk (Abstract is usually one chunk; Discussion 2–3). **Eyeball both.** Done.

## Role in the chain

The deepest **"why"** layer — the evidence base under the agent's reasoning: Rockwood = strike
*magnitude + spatial concentration*, Conn & Silber = the *speed-works* mechanism. Weakest agency
(enrichment, not a branch-driver), but it's what lets the agent answer the mechanism questions
and cite primary literature — and it's the source whose messy academic prose most justifies real
semantic search. Nice provenance: the agent can trace "the SAR says ~22 strikes/yr" → "because
Rockwood 2017 modeled it" → the actual Rockwood chunk.
