# Sanctuary Management Plan design (collection #3)

*Status: designed (2026-06-19). The third collection — owns the **mitigation / "what's being
done"** side of the CR↔Mgmt seam (`sanctuary-condition-collection-design.md`). Grounded in a
triage of all 5 files. Mirrors #2 (no card, `sanctuary` key, narrative) but adds a new lesson:
**the 5 files are heterogeneous in shape, and per-doc mapping absorbs it.** Parents:
`rag-collections.md`, `tool-design.md`, `scope-and-coverage.md`.*

## The finding — "management plans" are a mixed bag (triaged, not assumed)

The 5 files are *not* one template. Looked at them:

| File | Pages | What it actually is |
|---|---|---|
| `channel-islands-mgmt-2023` | **4** | §304(e) evaluation **memo** — the ship-strike mitigation gold |
| `olympic-coast-mgmt-2023` | **6** | short eval / summary |
| `cordell-bank-mgmt-2023` | **18** | short |
| `greater-farallones-mgmt-2023` | **28** | short |
| `monterey-bay-mgmt-2008` | **479** | the one **full** management plan (the outlier) |

A 4-page memo to a 479-page plan, mixed years and genres (forward-looking *plan* vs.
backward-looking *evaluation*). **There is no single section rule** — and that's fine: our
mechanic was already *per-doc hand-mapping on a tiny corpus*, so heterogeneity is absorbed by
treating each of the 5 individually. (Another "look at the data first" payoff — the assumption
that these were uniform large plans was wrong, and cheaply caught.)

For our question — *"is a ship-strike mitigation in force here?"* — both genres work: a memo
says "we did X (a VSR program slowed ships)", a plan says "we will do X." Either grounds the
agent's conclusion. In some cases the short **memos are better** — they're pre-compressed.

## The lane content — concentrated and small

The mitigation story is one tight slice. From the CINMS memo's **Resource Protection Action
Plan** bullet:

> *"…addressing the threat of ship strikes to whales, including adjusting shipping lanes…
> a Marine Shipping Working Group, and testing an incentive-based 'Blue Whales and Blue Skies'
> Vessel Speed Reduction program that has successfully slowed ships in the Santa Barbara
> Channel… apps (Spotter Pro and Whale Alert)…"*

That's the whole keep-set per doc: **Resource Protection / vessel-speed / ship-strike
mitigation** (VSR programs, lane adjustments, the working group, whale-alert apps). Drop the
other action plans (water quality, maritime heritage, operations, public awareness, education).

## Card? No *(consistent with #2)*

Narrative mitigation actions, not citable numbers. `sanctuary` is the join key (known upstream),
not a stored fact. Chunks only.

## Annotation

Join key **`sanctuary`**; `doc_type='mgmt-plan'` (covers both memos and the full plan — the
agent asks "what's being done here," indifferent to source genre); `section`; `source`; `year`;
`species` broad/null (a place, not a stock). **Contextual header:** `[Channel Islands NMS
Management Plan — Resource Protection / Ship-Strike Mitigation] <text>`.

## Mechanics — shape-specific, per doc

No template, so map each of the 5 to its own shape:
- **The four short 2023 docs (4–28 pp):** scan the **whole** doc, grab the relevant
  paragraph(s). No page-mapping needed — they're small.
- **Monterey 2008 (479 pp):** the only one needing CR-style **page-range selection** — find the
  Resource Protection action plan / ship-strike / vessel sections via the TOC, extract+clean
  only those pages.
- Recursive-split any over-long selection; **eyeball all 5**.

## Known coverage gap *(record it)*

**Monterey Bay's plan is 2008 — predating the modern VSR programs (mostly 2014+).** So for
Monterey (a humpback hotspot — where whale 479 lived), the corpus's "what's being done"
mitigation narrative is **dated/thin**: it won't describe today's vessel-speed measures. The
**SB-Channel (Channel Islands)** content is both the richest *and* the primary demo region, so
the agent should lean on CINMS for the mitigation story; Monterey is a known soft spot, not a
blocker. *(`vsr_zones` still carries the modern Monterey Bay NMS zone geometry — the gap is only
in the **narrative** of what's being done, not in the structured zone overlap.)*

## The payoff — the seam closes

#3 supplies the branch that **flips the F5 conclusion**: after the SAR says the species is
strike-vulnerable, the CR says the place is under traffic pressure, and `vessel_traffic` says
the whale's range overlaps the lanes — #3 answers *"…but is anything being done about it?"*
("yes — an incentive VSR program has measurably slowed ships in the SB Channel"). Severity
(SAR + CR + AIS) × mitigation (Mgmt) = the honest, non-alarmist risk verdict.
