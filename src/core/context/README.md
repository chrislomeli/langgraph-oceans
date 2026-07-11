# core.context

Drop-in context management for a LangGraph ReAct agent. It keeps long conversations
under control **without ever mutating the message log**: aged prose is summarized and
aged tool results are briefed into a small durable shadow state, and a compressed
**view** is assembled fresh for each model call.

- **Never mutates `messages`** — the raw log stays the single ground truth.
- **Self-contained** — one required dependency (a summarizer); everything else defaults.
- **Injectable seams** — swap the token counter or summarizer for tests or a custom setup.

---

## Quickstart

### 1. Construct a `ContextManager` (once, at startup)

```python
from core.context import ContextManager, ContextPolicy, LLMSummarizer

summarizer = LLMSummarizer(my_cheap_chat_model, my_summarizer_prompt)  # the one required dep

cm = ContextManager(
    summarizer,
    policy=ContextPolicy(
        chunk_token_budget=2000,     # summarize once this much prose has aged out
        live_tail_size=6,            # trailing messages kept raw, never summarized
        compaction_size_chars=1000,  # max chars kept of each tool result in its brief
        view_token_ceiling=None,     # set an int to hard-cap the assembled view
    ),
    # counter=...  # optional; defaults to the built-in HeuristicTokenCounter
)
```

### 2. Mix `ContextStateFields` into your graph state

```python
from core.context import ContextStateFields

class MyState(ContextStateFields, MyBaseState):
    ...   # adds: message_summaries, tool_calls, last_processed_message_id, token_count
```

Their reducers (`add_messages`, `update_dict`) are already declared — nothing else to wire.

### 3. Call `prepare(state)` in a node before the model runs

Updates the durable shadow state (summaries + tool briefs). Returns `{}` when nothing
has aged out yet.

```python
def summarize_node(state: MyState) -> dict:
    return cm.prepare(state)

graph.add_edge("summarize", "agent")
```

### 4. Call `build_view(state)` in the agent node to get what to send

Assembles the ephemeral view. Prepend your own (cache-stable) system prompt and invoke.

```python
def agent_node(state: MyState) -> dict:
    view = cm.build_view(state)                 # [manifest] + [summaries] + [tail]
    response = model.invoke([system_message] + view)
    return {"messages": [response]}
```

That's the whole surface: **construct, mix in, `prepare`, `build_view`.**

---

## What the two calls return

| Call | Returns |
|------|---------|
| `prepare(state)` | A durable-delta dict (`message_summaries`, `tool_calls`, `last_processed_message_id`) for your reducers — or `{}` when nothing aged out. **Never** returns `messages`. |
| `build_view(state)` | A `list[BaseMessage]`: a leading tool **manifest**, the inline **summaries**, then the raw **tail**. Ephemeral — recomputed every turn, never stored. |

---

## How it works (one paragraph)

`messages` grows forever and is never touched. A **bookmark**
(`last_processed_message_id`) marks how far back has been compressed. Each turn,
`prepare` checks whether a policy-budget's worth of prose has aged past the live tail;
if so it summarizes that prose into framed `HumanMessage`s and briefs its tool calls
into `ToolBrief`s, then advances the bookmark. `build_view` reassembles the outgoing
list from that shadow state + the raw tail. If you set `view_token_ceiling`, the view
is fit to that hard cap — shedding oldest briefs, then summaries, protecting the
current turn — **ephemerally**, so state is never affected. See `Summarization.md` for
the full design.

---

## Tuning — `ContextPolicy`

| Field | Meaning |
|-------|---------|
| `chunk_token_budget` | Target prose tokens per summarized slice (soft — shapes the steady state). |
| `live_tail_size` | Trailing messages kept raw and never summarized (soft). |
| `compaction_size_chars` | Max chars kept of each tool result in its brief. |
| `view_token_ceiling` | Hard cap on the assembled view. `None` = off. The backstop that makes the view always fit. |

**Several agents, one manager?** Pass `policies={"a": ..., "b": ...}` instead of
`policy=...`, then a `policy_key` on each call. A single `policy` needs no key.

---

## Swapping the seams (tests / custom setups)

The module depends outward only through two `Protocol`s, so you inject fakes or
alternatives without patching:

- **`TokenCounter`** — `count_messages` / `count_text`. Defaults to `HeuristicTokenCounter`
  (offline, approximate). Inject an exact tokenizer via `counter=`.
- **`Summarizer`** — `summarize(list[BaseMessage]) -> list[str]`. `LLMSummarizer` is the
  built-in adapter; any object with that method works.

See `tests/core/context/test_summarization.py` for a full deterministic suite (no LLM,
no DB) built on char-based / spy fakes at exactly these seams.
