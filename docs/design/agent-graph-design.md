# Design note — the agent graph (B5)

*Status: scaffolded 2026-06-20, tier-2 blanks open. Mid-level build spec for `src/agent/`.
Parent: `agent-orchestration-design.md` (the hub-and-zoom mental model + the tool-ad
template). This note is the **graph**: how the LLM, the bound tools, and the loop wire
together. Decisions marked **DECIDE** are yours (tier 3).*

## What it is

One LangGraph `StateGraph` running the **ReAct loop**: the LLM sees the question + tool
results so far, and either emits a tool call or a final answer. The "agency" is the LLM
deciding *which* tool to call next and *when to stop* — there is no scripted chain.

```
  START → agent ──(tool_calls?)──► tools ──► agent ──► … ──(no tool_calls)──► END
            ▲                                   │
            └───────────────────────────────────┘
```

- **agent node** — calls `llm.bind_tools(TOOLS).invoke([SystemMessage(SYSTEM_PROMPT)] + messages)`.
- **tools node** — LangGraph prebuilt `ToolNode(TOOLS)` executes whatever tool the LLM called.
- **the edge** — prebuilt `tools_condition`: if the last AI message has `tool_calls` → `tools`, else → `END`.

This is the canonical ReAct graph. The F5 flagship falls out of it: the LLM calls
`photo_id` → reads the `individual_id` → calls `sighting_lookup` → reads the `range_bbox`
numbers → calls `vessel_traffic`. The multi-hop seam is the LLM threading one tool's text
output into the next tool's args (which is why `sighting_lookup` returns the bbox as plain
numbers — see `agent/tools.py`).

## State

`MessagesState` (prebuilt: `messages: list` with the `add_messages` reducer). The message
list **is** the decision trace — every tool call and result is a message. A richer typed
`trace`/provenance field is a **B5-CTX** concern, deferred (build-status says extract it
from the vertical, not before).

## The tier-2 blanks (the learning — YOU write these)

1. **The tool ads** — the docstrings in `agent/tools.py` (`TODO(you)`). What the LLM reads
   to pick a tool. Without good ads it calls the wrong tool or none.
2. **`SYSTEM_PROMPT`** in `agent/graph.py` (`TODO(you)`) — the agent's brain: its scope
   ("conservation-risk, not general Q&A"), the hub-and-zoom decomposition reflex, when to
   **abstain** ("not in my sources"), and that it must cite. This is where the agent's
   behavior actually lives.

## DECIDE (tier 3 — your calls, defaults scaffolded)

- **Model** — default `claude-opus-4-8` (strongest reasoning for decomposition). Alt:
  `claude-sonnet-4-6` (cheaper/faster, very strong at tool use) if the ReAct loop iterates
  a lot. One-line swap (`MODEL` in `graph.py`).
- **Topology** — scaffold is the plain ReAct loop. Build-status names "**router → ReAct →
  recovery**". A **router** (classify the question first) and **recovery** (handle a tool's
  `ok=False`/`abstain`) are deferred extension nodes — add them when the plain loop's pain
  is real (same discipline as B5-CTX). DECIDE if/when.
- **`temperature`** — scaffolded at 0 (deterministic trajectories, easier to eval). DECIDE.
- **Image handling** — the agent does NOT see the photo; it gets the *path as text* and
  hands it to `photo_id`, which does the embedding. (Correct: the vision lives in the tool,
  not the LLM.) Revisit only if a question needs the LLM to actually look at an image.

## Build order

1. ✅ B-BIND — `agent/tools.py`, tools bound (ads = TODO(you)).
2. ✅ graph scaffold — `agent/graph.py` (system prompt = TODO(you)).
3. ✅ B-CLI — `agent/cli.py` (run it from the terminal).
4. **YOU:** write the 4 ads + the system prompt → `uv run python -m agent.cli "<question>" --image <path>`
   → watch it run the F5 chain. **That run is the B5 checkpoint.**
5. Then: tune ads/prompt against real trajectories; add router/recovery if needed; wire B7/B8 eval.
```
