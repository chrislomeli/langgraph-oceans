# Design note — the context layer (Layer 2.5)

> **Status: design, not built. ⚠️ EXTRACT THIS FROM THE SHIP-STRIKE VERTICAL — do not
> build it first.** Stand up one working agent (B5), feel the scratchpad sprawl / lost
> citations / blown token budget, *then* lift this layer out of what you actually needed.
> A context framework designed before an agent that hurts will abstract the wrong things.
>
> Mid-level — interfaces, responsibilities, boundaries, and build staging. Not code.
> Companion to `build-status.md` (item **B5-CTX**) and the agent design in
> `ocean-mammal-conservation-vision.md` (Layer 2).

## Why this exists (one paragraph)

The agent accumulates heterogeneous evidence across a multi-hop trace — image candidates,
a sighting range, traffic metrics, text chunks — each arriving as a typed `ToolResult`.
Something has to turn that growing pile into the **bounded, cited context the LLM sees each
turn**, and separately record **what happened** for audit/eval. That "something" is the
context layer. It's the unglamorous infra core that decides whether the agent is reliable;
most demos skip it, which is exactly why it's worth building as a first-class, reusable
framework. It is also the natural bridge for an infra engineer into agentic reasoning: you
cannot decide *what the model needs to see to make the next decision* without understanding
the decision.

## Where it sits — the seam

Layer **2.5**: between the dumb tools (Layer 3) and the LLM (Layer 5), serving the agent
loop (Layer 2). It owns the transformation `state + ToolResults → model-facing context`,
plus the structured trace.

```
Tools (L3) ──ToolResult──▶  CONTEXT LAYER (2.5)  ──rendered context──▶ LLM (L5)
                              │  state · assembly · compression
                              │  provenance · budget · trace
                              └──structured trace──▶ Layer-C eval + CLI render
```

## What it owns (six responsibilities)

1. **Working state (the run's memory).** The canonical typed object for one run: the
   `findings` scratchpad, `resolved_individual`, `candidates`, `iterations`, the trace.
   Owns the write moves (record a finding) and read moves (give me the scratchpad).
   *Single-run only; cross-session memory is explicitly out of scope.*

2. **Assembly / rendering.** Turn each heterogeneous result (`PhotoIDResult`,
   `VesselTrafficResult`, `SightingHistory`, `SearchResult`) into coherent text the LLM
   reads. The `ToolResult.summary` field is the seed; this layer decides ordering, include
   vs drop, and how a `range_bbox` from hop 2 stays legible at hop 4. *Hard part: faithful
   rendering of structured results into prose the model can reason over.*

3. **Compression.** When history outgrows the budget, fold older results into a running
   digest (the vision's `compress` node). *Hard part: compress without dropping facts still
   in play — the classic failure is summarizing away a value the agent still needs.*

4. **Provenance threading.** Every fact carries a `Citation`; this layer keeps the
   fact→citation binding alive **through assembly and compression**, so synthesis can ground
   every claim. *Hard part: compression is exactly where citations get orphaned. This is the
   backbone of "grounding is measured" — the one thing that must not leak.*

5. **Token budgeting.** The router sets a budget; this layer enforces it — measure context
   size, decide what to evict/compress, prioritize recent + load-bearing findings over stale
   ones. *Where infra instincts pay off most directly.*

6. **The trace (decision record).** A first-class structured log — route, tool order, hops,
   signals (`abstain`/`margin`/`ok`), dependency satisfaction, termination. **Distinct from
   the scratchpad:** scratchpad = what the LLM sees to decide *next*; trace = what happened,
   for *audit/eval*. Doubles as Layer-C eval input and the CLI "how I reasoned" render.

## What it does NOT own (boundaries — so it doesn't sprawl)

- **Not the tools** — they stay dumb, return `ToolResult` (Layer 3).
- **Not control flow** — it gives the agent what it needs to decide; the agent picks the
  next tool (Layer 2).
- **Not storage/indexes** — Layer 4. It assembles *results*, never touches pgvector.
- **Not vector mixing** — image and text stay separate in the index; this layer fuses their
  *results* in reasoning context, not the embeddings.
- **Not persistent/cross-run memory** — single session.

## The interface (the seam as an API)

```python
class Finding(BaseModel):
    claim: str                 # the rendered fact the model reads
    source: str                # which ToolResult produced it
    citation: Citation         # provenance, kept through compression
    hop: int                   # when it entered
    salience: float            # for eviction/prioritization under budget

class Step(BaseModel):         # one row of the trace
    node: str; tool: str | None; args: dict; signals: dict

class WorkingContext:
    findings: list[Finding]
    trace:    list[Step]
    budget:   TokenBudget

    def record(self, result: ToolResult) -> None      # ingest → findings + citations
    def render(self, role: str) -> str                # assemble model-facing context, in budget
    def compress(self) -> None                        # digest old findings, KEEP provenance
    def trace_step(self, node, tool, args, signals)   # append to the trace
    def citations(self) -> list[Citation]             # hand provenance to synthesis
```

Framework-grade and reusable across future agents (matches the standing "build once, reuse"
principle). It depends only on the `ToolResult`/`Citation` contracts (Layer 3), nothing below.

## Build staging — extract from pain, don't front-load

| Concern | First exercised by | When |
|---|---|---|
| state + assembly + **provenance** + **trace** | **ship-strike vertical** (multi tool results, `range_bbox` passed hop→hop, citations per tool, dependency for Layer C) | **MVP — build here** |
| **compression** | F4 RAG (text chunks flood the window) | later |
| **budgeting** | long traces / big chunk dumps | later |
| **retrieval fusion** (image + text + structured all return) | the full agent | later |

**MVP = state + assembly + provenance + trace**, lifted out of the ship-strike agent once
the scratchpad sprawl is real. Compression / budgeting / fusion arrive when RAG and the full
loop generate the pain that justifies them — which is also when the *right* abstraction is
knowable instead of guessed.

## The honest through-line

This is a real, framework-grade, infra-shaped component **and** it forces the agentic
reasoning, because "what does the model need to see to make the next decision" is
unanswerable without understanding the decision. That dual nature is the point — it leverages
infra strength while pulling toward the orchestration learning. The failure mode to watch:
letting it become "perfect the context store, reach the agent later." Built *after* B5,
around a real loop, it's the sweet spot; built first, it's infra-for-its-own-sake.
