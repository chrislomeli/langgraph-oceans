# Migration-prior reasoning — design note

> Mid-level design for the agent's "use what we know about humpback migration to fill
> the gaps the sparse data leaves." Written before code (component-design-first).
> Last updated: 2026-06-27.

## Why this exists

Two measured facts about our catalog force this:

1. **Per-individual data is sparse.** Median sightings per individual = **2**; 57% have ≤2;
   only ~10% have ≥10. You cannot characterize an individual's seasonal range from its own
   points.
2. **73% of humpbacks are seen in only ONE region** (mean 1.29 distinct regions/individual).
   So for three-quarters of whales we *never directly observe the migration link* — we see
   them feeding **or** breeding, never both.

Meanwhile the embedder caps at **reid@1 0.619** — the true individual isn't even rank-1 about
38% of the time, and close look-alikes are the common case (B0: genuine/impostor scores
overlap heavily).

The migration prior is how the agent makes a *correct, useful* statement anyway: it uses the
**one observed region + known migration structure** to infer the unobserved half of a whale's
range, and to demote candidates whose biology doesn't fit. It is a **plausibility prior, not
an oracle** — it modulates confidence and fills gaps; it never manufactures an identity.

## The knowledge it stands on (and where each piece lives)

This feature is a clean instance of an agent juggling three knowledge sources. Stating where
each lives *is* the design:

| Knowledge | Source | Why there |
|---|---|---|
| This whale's actual sightings / regions | **Tools** (`sighting_lookup`, `sighting_context`) | Ground truth the model cannot know. |
| Humpback migration structure (who breeds where, site fidelity) | **The model's own world knowledge** | Well-documented science already in Opus 4.8's weights. **We do NOT build a migration DB.** |
| *Reliably applying* the biology, and *bounding* it from over-claiming | **Scaffolding** — the domain brief below | Knowledge the model *has* ≠ knowledge it *uses*. The brief closes that gap and enforces hedging. |

The whole point: no new tool, no new data table. The prior rides on existing tool output plus
a prompt brief. (One small optional code touch — see "Build surface.")

## The migration structure (what the brief encodes)

North Pacific humpbacks have **multiple** breeding grounds, connected to feeding grounds with
strong **maternally-inherited site fidelity to the feeding area**. The discriminating axis:

- **California / Oregon feeders → winter in Mexico & Central America.**
- **N-BC / SE-Alaska feeders → winter in Hawaii.**

Our catalog spans the whole system, which is why this is usable here and not just in textbooks:

| region | share of humpback sightings |
|---|---|
| California (feeding) | 28% |
| Hawaii (breeding) | 24% |
| N-BC/SE-Alaska (feeding) | 23% |
| Mexico/CentAm (breeding) | 16% |
| OR/WA/S-BC (feeding) | 8% |
| Gulf of AK / other | ~2% |

Two feeding poles (CA, SE-Alaska), two breeding poles (Mexico, Hawaii). The prior is **strong
but probabilistic** — there is real exchange; it is a population-level tendency, not a law.

## When the agent invokes it

**Trigger A — disambiguation.** `photo_id` returns close candidates (thin `margin`). The agent
pulls each candidate's region(s). *Only if the candidates sit in different feeding regions* does
the prior help: combine each candidate's likely full range with the query photo's location to
**demote** the candidate whose biology doesn't fit.

**Trigger B — range / exposure completion.** A (possibly confident) ID has sparse, single-region
sightings. The agent infers the unobserved half: "seen only feeding off CA → likely winters in
Mexico." This adds correct information the user couldn't derive, and it flags that part of the
animal's year sits **outside our West-Coast AIS coverage** (honest boundary for the F5 risk read).

## When it must NOT fire (false-positive guards)

- **Candidates in the same region** → biology can't separate them. Say so; do not invent a
  difference.
- **Species other than humpback.** The brief is humpback-specific. Gray whales, orcas, etc. have
  different patterns — the agent must **not** apply humpback migration to them. (~90% of the
  catalog is humpback; the other 10% is the trap.)
- **Direct observation always wins.** If an individual *actually has* a sighting in region X, that
  is fact; the prior only fills *gaps*, never overrides an observed point.
- **The prior may DEMOTE, but rarely PROMOTE alone.** It can lower a candidate's plausibility; it
  should almost never be the *sole* reason to confirm an ID.

## Output discipline (this is what keeps false positives near zero)

1. **Directional language only.** "more consistent with," "less plausible," "likely winters in" —
   never "definitely," "confirmed," "is a Hawaii whale."
2. **Label inference vs. observation.** Distinguish "seen off Monterey (observed)" from "likely
   winters in Mexico (inferred from migration pattern)."
3. **State the basis** whenever the prior is used: which observed region, which structural fact.
4. **Always allow abstention.** "Two candidates, both off California — migration can't separate
   them; here's what each is" is a *success*, not a failure.

## Build surface (kept deliberately small)

- **The domain brief** — a short paragraph appended to / merged into `SYSTEM_PROMPT`
  (`agents/sandbox_agent/graph.py`), encoding the structure + the two triggers + the hedging
  rules above. This is the whole feature.
- **(Optional, recommended) coarse region labels in tool output.** Have `sighting_lookup` /
  `sighting_context` emit a named migratory region (CA / PNW / SE-AK / Hawaii / Mexico) per
  sighting, so the agent reasons over clean tokens instead of raw lat/lon (where it could
  mis-bucket). Small, contained change; lowers the chance of a geography slip.
- **No new tool, no migration table, no model fine-tune.**

## How we'll know it works

Headline metric: **near-zero false positives** — the agent never confidently wrong *because of*
a migration over-claim. Demonstrated with a small curated trajectory set (B8), one per branch:

1. Two candidates, **different** feeding regions, query location resolves → correct demotion, hedged.
2. Single sparse ID → correct hedged range/exposure completion.
3. Two candidates, **same** region → agent correctly *declines* to use the prior.
4. A **gray whale** (or other non-humpback) → agent does **not** apply the humpback brief.

A few good, honest traces is the bar — not coverage of all 23.7k whales.

## Open questions (deferred)

- Do we hand the agent the region-label mapping (the optional build above), or trust it to bucket
  lat/lon itself? Lean: give it labels.
- Where exactly does the brief live — inline in `SYSTEM_PROMPT`, or a separate
  `prompts/` entry the brief is composed from? (Ties into the future B5-CTX context-assembly idea.)
- Does the same pattern generalize to gray whales later (coastal CA→Mexico migration), or stay
  humpback-only? Out of scope for now.
