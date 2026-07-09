# core.context — caller's-eye view

How an external caller wires and uses the context-management module. Companion to
`Summarization.md` (the spec) and the `src/core/context/` scaffold.

A caller only ever touches **three seams**:

1. **Construct** `ContextManager` once, at the composition root (inject its collaborators).
2. **Call** `context.prepare(state)` from a thin graph node each turn.
3. **Register** the `get_full_tool_result` tool so the agent can dereference compacted results.

Everything else (chunking, compaction, summarizing, view assembly) is internal and
never called directly.

---

## Component diagram

```
                         ┌──────────────────────────────────────────┐
  COMPOSITION ROOT       │  src/app/context.py   build_context()     │
  (wires once, startup)  └───────────────────┬──────────────────────┘
                                             │ constructs + injects
                                             ▼
  GRAPH  (the caller)    ┌──────────────────────────────────────────┐
  agents/sandbox_agent   │  summary_node(state)  ── thin wrapper ──┐ │
                         │  agent_node(state)   (runs on the view) │ │
                         └─────────────────────────────────────────┼─┘
                                                                   │ .prepare(state)
                                       ┌───────────────────────────▼───────────────┐
  FACADE                               │              ContextManager                 │
  core/context/manager.py              │   prepare(state) -> state-update dict        │
                                       └──┬────────┬──────────┬──────────────┬───────┘
                                          │        │          │              │
                     ┌────────────────────┘        │          │              └──────────────┐
                     ▼                              ▼          ▼                             ▼
                chunking.py                   compaction.py  summarizer.py            view_builder.py
                next_boundary                 compact_...    summarize_chunk          build_view
                validate_no_orphans           should_compact
                     │                              │          │
                     └──────── use ───────┐         │          │
                                          ▼         ▼          ▼
  INJECTED PROTOCOLS               TokenCounter  ToolResultStore  Summarizer
  core/context/protocols.py             ▲            ▲              ▲
                                        │ satisfies  │ satisfies    │ satisfies
  CONCRETE IMPLS               HeuristicTokenCounter  InState...   LLMSummarizer
  (structural, no import back)  core/llm/token_counter compaction  core/context/summarizer
```

Read the arrows as "depends on / calls." The three protocols are the only places
the module reaches outward, and the concrete impls satisfy them *structurally* —
they never import the protocol. That is what keeps tiktoken, your LLM registry,
and any storage backend out of the module.

---

## Seam 1 — construct once (composition root)

```python
# src/app/context.py — inside build_context(), alongside your existing deps
from core.llm.token_counter import HeuristicTokenCounter
from core.context import ContextManager, ContextPolicy
from core.context.summarizer import LLMSummarizer

context_manager = ContextManager(
    counter=HeuristicTokenCounter(),  # core/llm — the salvaged counter
    summarizer=LLMSummarizer(llm_registry.get("summary")),  # a DISTINCT cheap model, not the agent's
    config=ContextPolicy(
        chunk_token_budget=2000,  # size of one summarized chunk
        live_tail_size=6,  # trailing messages never chunked / never compacted
        compaction_size_chars=1000  # tool results smaller than this are left full
    ),
)
# stash on AppContext.deps so the graph builder can reach it
```

Note: the `ToolResultStore` is NOT injected here. It's per-turn — backed by the
`tool_result_refs` dict that lives in graph state — so `prepare` builds it from
state each call. Construction only needs the counter, the summarizer, and config.

## Seam 2 — call each turn (the graph node)

The node is a 3-line wrapper. All the work is behind `prepare`.

```python
# agents/sandbox_agent/graph.py
def summary_node(state: OceanState) -> dict:
    return context_manager.prepare(state)      # returns a state-UPDATE dict

graph.add_node("summarize", summary_node)
graph.add_edge("summarize", "agent")           # runs right before the model call
```

What `prepare(state)` does and returns:

```
prepare(state):
    reads   : messages, summary_chunks, tool_result_refs, last_processed_message_id
    ─────────────────────────────────────────────────────────────────────────────
    1. delta = tokens of messages AFTER the bookmark          # cheap, real content
    2. if delta < chunk_token_budget:
           view = build_view(...)                             # unified path; may == raw
           return {"llm_input_messages": view}                # no durable change
    3. else:
           b       = next_boundary(messages, bookmark, ...)   # tool-pair-safe slice
           compact any aged tool msgs in b  -> tool_result_refs updates
           chunk   = summarize_chunk(b, ...)                  # structured summary
           view    = build_view(messages, chunks + [chunk], new_bookmark)
           return {
               "llm_input_messages": view,                    # EPHEMERAL — for this call only
               "summary_chunks": [chunk],                     # append reducer
               "tool_result_refs": {...},                     # merge reducer
               "last_processed_message_id": new_bookmark,     # advance the bookmark
           }
```

The one rule to remember: the return **never** contains `"messages"`. The ground-truth
log is untouched; `llm_input_messages` is the ephemeral view LangGraph uses for the
single model call and then discards.

## Seam 3 — register the dereference tool

Compacted tool results carry a `ref_...` pointer. Give the agent a way to pull the
full payload back when it actually needs it.

```python
from core.context.compaction import build_get_full_tool_result_tool

TOOLS = [ ...your 4 tools..., build_get_full_tool_result_tool() ]
# the tool reads state["tool_result_refs"][ref_id] at call time (InjectedState)
```

Escape hatch, not the default read path — dereferencing re-injects the full payload
as a fresh ToolMessage, which is itself a future compaction candidate.

---

## The full turn, end to end

```
   state.messages (grows forever, never mutated)
          │
          ▼
   summary_node ──▶ context.prepare(state)
          │              │
          │              ├─ under budget ─▶ view = summaries + raw tail
          │              └─ over  budget ─▶ chunk+compact+summarize, then view
          ▼
   {"llm_input_messages": view, ...shadow-state updates...}
          │
          ▼
   agent_node runs the model on `view`  (never sees the full log)
          │
          ▼
   model output ──▶ add_messages appends to state.messages  (log grows; loop repeats)
```
