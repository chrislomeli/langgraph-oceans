# Scope & coverage — what this agent is for, and what it can actually answer

*Status: scope decision (2026-06-19). The boundary that keeps us out of the "general whale
chatbot" corner. Read this before adding a data source or designing a tool — it's the test
for whether something is in scope. Companion to `rag-collections.md` (the collections) and
`build-status.md` (live status).*

## The lane commitment

**This is a conservation-risk reasoning agent — not a general whale Q&A bot.**

Its job: *given a whale photo and a question, identify the individual, then reason about its
**conservation situation and risk** by chaining across what's known about the animal, its
population, and the water it lives in — and cite the sources.*

The temptation that sinks the project is scope-creeping into "ask me anything about whales."
We can't cover that, and chasing it makes the demo look thin. The discipline: **commit to the
conservation-risk lane.** Inside it, coverage is genuinely strong. Outside it, the agent says
*"that's not in my sources"* — and that honesty (abstention + citations) is a feature, not a
failure.

## The user never sees a menu

A real user asks one broad thing — *"tell me about this whale"* — and won't know to ask the
precise sub-questions. That's fine: **the agent decomposes the broad question.** The tool
"menu" is internal — it's how the *agent* matches its own sub-goals to data, not something the
user must learn. The user asks once; the agent runs the chain and composes the answer.

## What the data can tell the agent — three levels of zoom

Each collection answers at a different level. The plain-language version:

### Level 1 — *this whale* (per-individual / per-encounter)
| Collection | What it tells the agent | Why the user cares |
|---|---|---|
| `photo_id` + `individuals` | Who it is, its name, have we seen it before | "Who is this? Have we met it?" |
| `sighting_lookup` *(sightings)* | Where & when it's been seen; its home range | "Where does it go?" |
| `obis_seamap_points` *(oceano)* | The water at each sighting — depth, temperature, ecosystem, **which sanctuary**, who saw it, group size | "What kind of place does it live in? Is that protected water? Why is it there?" |

### Level 2 — *its kind* (per-species / stock)
| Collection | What it tells the agent | Why the user cares |
|---|---|---|
| **SAR** | How many, recovering?, what kills them, sustainable?, ship-danger level | "Is it endangered? What's the threat?" |
| Sanctuary condition / mgmt | What pressures the area faces, what's being done | "Is anyone protecting them?" |
| Reviews | Why this species is strike-prone; does slowing ships help | "Why are they at risk? Does anything help?" |

### Level 3 — *the water* (per-place)
| Collection | What it tells the agent | Why the user cares |
|---|---|---|
| **AIS** | How busy with ships a patch of ocean is, by year | "Is my whale swimming through a shipping highway?" |
| `vsr_zones` | Whether ships are *supposed* to slow down there | "Are there protections in this spot?" |

## The headline answers are chains, not lookups

No single collection answers the real questions — the agent **composes across levels.** This
is the whole "agentic" thesis. The flagship, worked out:

> **"Is this whale at risk from ships?"**
> ```
> sightings           where it is (its range)
>   × AIS             is that water busy?
>   × vsr_zones       is it a managed/slow-down zone?
>   × SAR             does this species actually die from strikes?  + the ~10% detection caveat
>   × obis/oceano     depth + season = the risk window
> ```
> AIS alone is meaningless — it knows nothing about whales. The SAR alone is generic. Only
> the **chain** answers the question.

> **"Tell me about this whale"** = identity + range + conditions + population + threats +
> protection + ship-risk — the agent fans out across all three levels and writes one paragraph.

**Implication:** coverage must be judged at the **answer/chain level**, not per collection.

## Honest coverage

Across all three levels, the same line holds:

- ✅ **Covered well:** identity · re-ID · location/range · environmental conditions · population
  status · threats & mortality · protection measures · ship-strike risk.
- ❌ **Not covered:** health/injury of this individual · age · size · family/kinship/calves ·
  diet or behavior of *this* animal · anything real-time (we have historical sightings, not
  live tracking).

So of the *full* curiosity space we cover maybe half the categories — but the half we cover
**is the conservation-risk lane**, which is the product. The other half is individual biology,
which no corpus we can realistically assemble would reach. We say so out loud and the agent
declines gracefully, rather than papering over it.

## The test (use this going forward)

When considering a data source or a tool: **does it strengthen a chain inside the
conservation-risk lane?** If yes, it's in scope. If it only serves "general whale facts," it's
out — note it and move on. This is the rule that prevents the corner.
