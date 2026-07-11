# core.context — caller's-eye view

How an external caller wires and uses the context-management module. Companion to
`Summarization.md` (the spec) and the `src/core/context/` source.

The module keeps a durable **shadow state** (summaries + tool briefs) derived from the
raw message log, and assembles a compressed **view** for each model call. The log
itself is never mutated.

A caller touches **three seams**:

1. **Construct** `ContextManager` once, at the composition root (inject the token counter + policies).
2. **`prepare(state, policy_key, summarizer)`** from a thin summary node each turn — updates the durable shadow state.
3. **`build_view(state, policy_key)`** from the agent node each turn — assembles the ephemeral view to send the model.

Everything else (chunking, compaction, summarizing, fitting, manifest rendering) is
internal and never called directly.

---

## Component diagram

```
                         ┌──────────────────────────────────────────┐
  COMPOSITION ROOT       │  src/app/context.py   build_context()     │
  (wires once, startup)  └───────────────────┬──────────────────────┘
                                             │ constructs + injects
                                             ▼
  GRAPH  (the caller)    ┌──────────────────────────────────────────┐
  agents/sandbox_agent   │  summary_node(state) ─ prepare(...)        │
                         │  agent_node(state)   ─ build_view(...)      │
                         └───────────────────┬─────────────────────┬─┘
                                 .prepare(...)│                     │.build_view(...)
                                       ┌──────▼─────────────────────▼──────┐
  FACADE                               │            ContextManager          │
  core/context/manager.py             │  prepare -> durable delta dict      │
                                      │  build_view -> list[BaseMessage]     │
                                      └──┬────────┬──────────┬──────────────┘
                                         │        │          │
                     ┌───────────────────┘        │          └──────────────┐
                     ▼                            ▼                          ▼
                chunking.py                 compaction.py             view_builder.py
                next_boundary               brief_tools               build_view / _fit_* /
                (the gate)                  (tool briefs)             _render_manifest
                     │                            │                          │
                     └──────── use ──────┐        │  summarizer.py           │
                                         ▼        ▼  summarize_messages       ▼
  INJECTED PROTOCOLS               TokenCounter        Summarizer      (uses TokenCounter)
  core/context/protocols.py             ▲                  ▲
                                        │ satisfies        │ satisfies
  CONCRETE IMPLS               HeuristicTokenCounter    LLMSummarizer
  (structural, no import back)  core/llm/token_counter   core/context/summarizer
```

Read the arrows as "depends on / calls." The protocols are the only places the module
reaches outward, and the concrete impls satisfy them *structurally* — they never
import the protocol. That keeps tiktoken, your LLM registry, and any storage backend
out of the module.

> `ToolResultStore` (protocols.py) and `retention.py` are a **parked seam** for the
> MEMORY axis (durably shrinking the checkpoint). Not wired; ignore for context use.

---

## Seam 1 — construct once (composition root)

`ContextManager` takes the token counter and a map of named policies. The summarizer
is NOT injected here — it's passed per-call to `prepare` (so a graph can use different
summarizer models per policy).

```python
# src/app/context.py — inside build_context(), alongside your existing deps
from core.llm.token_counter import HeuristicTokenCounter
from core.context import ContextManager, ContextPolicy

context_manager = ContextManager(
    counter=HeuristicTokenCounter(),          # core/llm — the salvaged approximate counter
    policies={
        "oceans_agent": ContextPolicy(
            chunk_token_budget=2000,          # target prose tokens per summarized slice (SOFT)
            live_tail_size=6,                 # trailing messages never summarized (SOFT)
            compaction_size_chars=1000,       # max chars kept of each tool result in its brief
            view_token_ceiling=None,          # HARD ceiling; None = fitter OFF
        ),
    },
)
# stash on AppContext.deps so the graph builder can reach it
```

A missing `policy_key` fails loud (no silent fallback) — see `ContextManager._policy`.

## Seam 2 — update the shadow state each turn (summary node)

A thin node that builds the summarizer and calls `prepare`. `prepare` gates on
`next_boundary`: it returns `{}` when nothing has aged out yet, otherwise the durable
deltas.

```python
# agents/sandbox_agent/graph.py
summarizer = LLMSummarizer(llm_registry.get("summarizer"), system_prompt)  # a DISTINCT cheap model

def summary_node(state: OceanState) -> dict:
    return context_manager.prepare(state, policy_key="oceans_agent", summarizer=summarizer)

graph.add_edge("summarize", "agents")   # runs right before the agent node
```

What `prepare` reads and returns:

```
prepare(state, policy_key, summarizer):
    reads   : messages, tool_calls, last_processed_message_id
    ────────────────────────────────────────────────────────────────────────────
    b = next_boundary(messages, bookmark, ...)          # the gate
    if b is None:  return {}                              # nothing aged out; no change
    else:
        summaries = summarize_messages(b, summarizer)    # prose -> framed HumanMessage(s)
        briefs    = brief_tools(b, base_ordinal=len(state.tool_calls))   # tools -> ToolBrief(s)
        return {
            "message_summaries":          summaries,     # add_messages reducer
            "tool_calls":                 briefs,        # update_dict reducer
            "last_processed_message_id":  b.end_message_id,   # advance the bookmark
        }
```

The one rule: the return **never** contains `"messages"` — the ground-truth log is
untouched. `prepare` also never returns the view.

## Seam 3 — assemble the view each turn (agent node)

`build_view` recomputes the ephemeral view from the shadow state. The agent node
prepends its own (cache-stable) system prompt and invokes the model on the result.

```python
# agents/sandbox_agent/graph.py
def agent_node(state: OceanState) -> dict:
    view = context_manager.build_view(state, policy_key="oceans_agent")
    response = llm.invoke([SystemMessage(system_prompt)] + view)
    return {"messages": [response]}         # add_messages appends; the log grows
```

The view is `[manifest] + [summaries] + [tail]`:

- **manifest** — the tool briefs as ONE leading framed `HumanMessage` (dedup index).
- **summaries** — the cold region, inline, at normal salience.
- **tail** — everything after the bookmark, raw.

Passing `policy_key` activates the HARD ceiling: when the policy sets
`view_token_ceiling`, `build_view` fits the three layers to budget (tail protected,
then summaries, then briefs shed) — ephemerally, never touching state. Omit
`policy_key` (or leave the ceiling `None`) and every layer is kept.

---

## The full turn, end to end

```
   state.messages (grows forever, never mutated)
          │
          ▼
   summary_node ──▶ prepare(state, ...)
          │              ├─ nothing aged out ─▶ {}                (no durable change)
          │              └─ aged out ─────────▶ summaries + briefs + advanced bookmark
          ▼
   ...durable shadow-state updates merged by reducers...
          │
          ▼
   agent_node ──▶ view = build_view(state, ...)   # [manifest] + [summaries] + [tail]
          │        model runs on [system] + view  (never sees the full log)
          ▼
   model output ──▶ add_messages appends to state.messages  (log grows; loop repeats)
```
