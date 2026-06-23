# Where we are — working notes

> A "pick it up later" note to keep us both oriented, not a system prompt. When this
> disagrees with reality, fix this first. Last updated: 2026-06-22.

## What we're doing right now

- **Coming up to speed on the chat interface** — the streaming front-door to the agent.
- Two new files are sitting in the repo **root** and **need to be moved** once we're done
  poking at them:
  - `src/chat_app.py` — the streaming backend (`ocean_runner`): the ONE seam between the
    LangGraph graph and the `agent_chat` chat UI.
  - `src/debug_runner.py` — a no-server driver that calls `ocean_runner` directly so we
    can set breakpoints and watch frames without HTTP/SSE.
- **Chris is running + debugging `debug_runner.py`** to get a feel for this simple flow
  before adding more functionality. (One-step-at-a-time — don't bolt on features yet.)

## The flow we're studying

```
debug_runner.main()
  → ocean_runner(TurnRequest)            # src/chat_app.py
      → GRAPH.astream_events(...)        # the sandbox_agent ReAct graph, built once
          START → agents → tools → agents → … → END
      → yields Token / ToolCall frames   # translated from graph events
```

- `astream_events` **streams**: we handle a flow of fine-grained events, not one return
  value. The `async for` ends on its own when the graph hits `END` — we don't decide when
  to stop. The `if/elif` is a **classifier**, not a collector: it forwards the 3 event
  kinds a chat UI cares about (`on_chat_model_stream`→`Token`, `on_tool_start`/
  `on_tool_end`→`ToolCall`) and ignores the rest.
- There's a temporary **`[event] …` debug print** in `ocean_runner` that logs *every*
  event so we can watch one turn's full sequence. **Remove it** (or gate behind an env
  var) before this goes anywhere real — it also fires under uvicorn.

## How to run

```bash
# direct, breakpoint-friendly (what Chris is doing now)
uv run python src/debug_runner.py

# full streaming server + React front end
AI_ENV_FILE=.env uv run uvicorn chat_app:app --reload --app-dir src

# single-shot CLI (no streaming)
uv run python -m agents.sandbox_agent.cli "<question>" --image PATH --trace
```

## Map of the agent code (recent restructure, not yet committed)

- `src/agents/sandbox_agent/` — the ocean agent: `graph.py` (ReAct loop, `SYSTEM_PROMPT`,
  Opus 4.8), `cli.py`.
- `src/agents/tools.py` — the 4 LLM-callable tools + their "ads" (docstrings the model
  reads to pick a tool). photo_id, sighting_lookup, sighting_context, vessel_traffic.
- `src/agents/commons/` — **generic agent infra being ported in from another project**
  (docstrings still say "world-simulator / cluster agents / supervisor"). Currently wired
  via `@node_executor("agent_node")` for metrics + error handling. Note: the graph uses
  plain `MessagesState` (no `session_id`), so the session-tracing half is inert for now.

## Loose ends / when we pick this up

- [ ] Move `chat_app.py` + `debug_runner.py` out of root into a sensible home.
- [ ] Remove (or env-gate) the `[event]` debug print in `ocean_runner`.
- [ ] Decide what `commons/` we actually keep vs. drop from the ported world-simulator
      layer; reconcile its `TracedState`/`session_id` with the graph's `MessagesState`.
- [ ] None of the agent restructure is committed yet — `src/agent/` → `src/agents/`.

## Where we're heading (next, after we're comfortable with the flow)

Once the simple chat flow is solid, the two things we want to add:

1. **More complex use cases** — beyond the single F5 ship-strike chain: multi-hop
   reasoning, disambiguation (F3), grounded doc answers (F4), etc.
2. **"Command" actions** — like Claude's `/skills`, but aimed at **declaring what to
   pull into context for a given request**: a named, reusable bundle that says "for
   *this* kind of question, load *these* tools / data / instructions." It's context
   *assembly*, not a procedure to run. This is the `/skills`-shaped front end to the
   **B5-CTX context layer** (state · assembly · provenance) already on the roadmap —
   the `commons/` infra we're porting in is the substrate for it.

## Status of the bigger build

The live dashboard is `docs/build-status.md`. Short version: F1 (photo-ID) done; **F5
ship-strike is agentic** (the LLM chains photo_id → sighting_lookup ∥ sighting_context →
vessel_traffic itself); tool ads + system prompt are written. Next big rocks: doc_search /
stock_facts tools, router/recovery nodes, eval-driven tuning (B7/B8).
