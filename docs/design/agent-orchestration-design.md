# Agent orchestration — how we advertise the data to the agent

*Status: design in progress (2026-06-19). How we describe the tools + frame the task so the
agent actually **chains across collections** (promotes usage) instead of under-calling. Feeds
the B5 agent's system prompt + tool descriptions. Built on `scope-and-coverage.md` (the lane +
three levels) and `rag-collections.md` (the collections).*

## The problem we're solving

We chose **small tools (one per collection) + an orchestration recipe** over a scripted
composite tool — to keep genuine agency (the thing the eval scores). The risk that buys us:
**under-chaining.** Opus 4.8 (our model) is documented to *under-reach* for tools by default —
left alone, it'll answer "tell me about this whale" from `photo_id` + its own knowledge and
never touch SAR or AIS. The orchestration design is how we beat that without scripting the
agent into a non-agent.

Two failure modes to design against:
- **Under-use / under-chain** ← the main target of "promote usage."
- **Mis-use / hallucinate outside the lane** ← handled by per-tool anti-overlap + gap honesty.

---

## 1. The mental model — hub-and-zoom *(LOCKED 2026-06-19)*

The frame we put in the agent's head:

> **The individual whale is the hub. You reason by zooming out from it along three spokes —
> and the answers that matter live at the *intersections*, not on any single spoke.**

- **Anchor the individual first.** Start with *"who is this?"* — identity unlocks everything
  (its own history *and* which population it belongs to). No anchor, no chain.
- **Three zooms from the hub:**
  - **IN → its life:** where/when it's been seen, in what waters *(this whale)*
  - **OUT → its kind:** population status, threats, protections *(the species)*
  - **DOWN → its waters:** how busy / how managed the places it frequents are *(the ocean)*
- **Meaning lives at the intersections.** The agent's reflex is *"what do I need to **connect**
  to answer this?"* — never *"which single tool answers this?"* (Ship-strike risk = the
  individual's locations × the species' strike-vulnerability × how busy those waters are.)
- **Anchor, then expand to the goal.** ID first; pull all spokes for a broad question, a
  subset for a pointed one.
- **Claim only what the spokes support.** Missing spoke (health, age, family) → say so.

**Why this shape:**
1. **Mirrors the data architecture on purpose** — `individual_id` is already the relational hub
   bridging the vector spaces (vision doc); the mental model is that hub made *cognitive*, so
   the agent's reasoning and the data wiring never fight.
2. **Principles, not a script** — deliberately *not* "always call A→B→C"; a fixed sequence
   would kill the agency the eval is meant to measure. It's a *way of thinking*, applied with
   judgment.
3. **Directly attacks under-chaining** — "meaning lives at the intersections" is the exact
   instruction that stops single-tool-plus-LLM-knowledge answers.

**Calibration *(LOCKED)*:** lean **prescriptive on one thing only — the "anchor first, then
ask what to connect" reflex** — and keep everything else judgment. That single nudge reliably
triggers the fan-out without scripting the order. Too prescriptive → script (kills agency);
too loose → under-chaining. The anchor-and-connect reflex is the one place we push.

---

## 2. Per-tool advertising — "call this when…" + anti-overlap  *(template LOCKED; ads live in tool-design.md)*

Each tool's ad has a locked three-part template: **Trigger** ("call this when the goal is X" —
Opus 4.8 responds strongly to explicit triggers) + **Anti-overlap** ("not for Y — that's the
\_\_ spoke") + **Returns/level**. Two properties we designed in: every "not for…" points at the
neighbor (seams close), and each tool's "keyed on `individual_id`/`species`/`range_bbox`" line
re-teaches the hub-and-zoom dependency (no spoke without the anchor).

The **filled-in ads now live in `tool-design.md`** as the agent-facing facet of each tool
(alongside its Query + Storage), so the per-tool contract is designed in one place.

## 3. Gap advertising — honest "not in my sources"  *(NEXT)*

How the agent is told the out-of-lane gaps (health/age/family/behavior/real-time) so it
abstains gracefully instead of hallucinating. To design.

## 4. Encoding — system prompt vs. tool descriptions  *(LATER)*

Where each piece lives (the mental model in the system prompt; triggers in tool descriptions;
etc.) and how it interacts with prompt caching. To design once 1–3 settle.
