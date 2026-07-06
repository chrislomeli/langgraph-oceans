# Where we are — working notes

> A "pick it up later" note to keep us both oriented, not a system prompt. When this
> disagrees with reality, fix this first. Last updated: 2026-07-06.

## What we're doing right now

- **Arc A (the injection spine) is DONE and verified end-to-end** (2026-07-06). The
  `GRAPH = build_graph()` global is gone; everything the runner needs arrives via a context
  object built once at startup. **NEXT: Arc B / B1 — the checkpointer** (multi-turn memory).
  The saver choice (B1.1) is a live discussion, not yet decided.
- The two entry points live in the **`src/app/`** shell:
  - `src/app/ocean_runner.py` — the streaming backend: the ONE seam between the LangGraph
    graph and the `agent_chat` chat UI. Holds `ocean_runner(request, ctx)` + the `OceanRunner`
    closure class + the server `app`/`lifespan`.
  - `src/app/debug_driver.py` — a no-server driver that builds `ctx` by hand and calls
    `OceanRunner(ctx)` directly so we can set breakpoints and watch frames without HTTP/SSE.
  - `src/app/context.py` — the composition root: `build_context() -> AppContext(settings,
    graph, deps)`, built once at startup, never at import.
- Task list: `docs/design/work-breakdown.md` (Arc A done; Arc B is the portfolio work).

## The flow we're studying

```
debug_driver.main()
  → build_context() -> AppContext         # src/app/context.py (settings, graph, deps)
  → OceanRunner(ctx)(TurnRequest)         # src/app/ocean_runner.py
      → ctx.graph.astream_events(...)     # the sandbox_agent ReAct graph, built once
          START → agents → tools → agents → … → END
      → yields Token / ToolCall frames    # translated from graph events
```

- `astream_events` **streams**: we handle a flow of fine-grained events, not one return
  value. The `async for` ends on its own when the graph hits `END` — we don't decide when
  to stop. The `if/elif` is a **classifier**, not a collector: it forwards the 3 event
  kinds a chat UI cares about (`on_chat_model_stream`→`Token`, `on_tool_start`/
  `on_tool_end`→`ToolCall`) and ignores the rest.

## How to run

```bash
# direct, breakpoint-friendly (needs AI_ENV_FILE exported → the SECRETS/.env)
uv run python -m app.debug_driver

# full streaming server + React front end
AI_ENV_FILE=.env uv run uvicorn app.ocean_runner:app --reload --app-dir src
```

## Map of the code — three tiers (restructured 2026-07-03, committed 2026-07-06)

Layout follows `docs/design/app-context-design.md`: **core → domain → app shell**, with the
rule that `core/` imports nothing domain-specific.

- `src/core/` — **PLATFORM** (domain-agnostic; the generic substrate):
  - `config.py`, `exceptions.py`, `logging_config.py`
  - `core/llm/` (LLMRegistry), `core/prompts/` (PromptRegistry)
  - `core/agents/` — generic agent-graph framework: `node_executor`, `node_types`
    (NodeError + TracedState), `node_metrics`, `state_types`, `routing` (unused), and
    `dependencies.py` (`AgentDependencies` container — the app-context shape). Docstrings
    still say "world-simulator" in places — cosmetic cleanup pending.
- `src/agents/` — **DOMAIN** (the ocean agent): `sandbox_agent/graph.py` (ReAct loop on
  `OceanState`, Opus 4.8; the LLM is `llm_registry.get(role).bind_tools(TOOLS)` — binding
  lives in the graph builder, not the registry), `tools.py` (the 4 LLM-callable tool ads;
  the system prompt now lives in `core/prompts/templates/oceans_agent/v1/`). Also domain:
  `tools/`, `models/`, `stores/`, `rag/`. (`sandbox_agent/cli.py` was deleted — the two
  `app/` entry points supersede it.)
- `src/app/` — **SHELL** (entry points): `context.py` (composition root `build_context()`),
  `ocean_runner.py` (server + runner seam), `debug_driver.py` (no-server driver).

## Loose ends / when we pick this up

- [x] Move `chat_app.py` + `debug_runner.py` out of root → now in `src/app/` (2026-07-03).
- [x] Remove the `[event]` debug print in `ocean_runner` (deleted 2026-07-03 with `Colors`).
- [x] Weed `commons/`: deleted `risk_view.py` + `schemas.py` (dead wildfire code);
      rescued `TracedState` → `node_types.py`. Survivors are all generic infra now:
      node_executor, node_types (NodeError+TracedState), node_metrics, state_types,
      agent_dependencies, routing (route_base — KEEP as future infra, currently unused).
      These are the set earmarked for `core/agents/` when the `core/` move happens.
- [x] Reconciled `TracedState` with the graph state (2026-07-03): the graph now runs on
      `OceanState(TracedState)` — a pydantic state = the traced fields (session_id, status,
      error) + a `messages` channel (`add_messages` reducer). `node_executor`'s metrics/
      error handling now have real channels to write to (verified: reducer appends AND
      status/error/session_id persist; ToolNode + tools_condition accept the pydantic state).
      Then: seed `session_id` from `TurnRequest` at invoke so tracing is populated.
- [x] Seeded `session_id` from `TurnRequest` into the graph state (2026-07-03): `ocean_runner`
      passes it in the input dict → `node_executor` stamps it on metrics + NodeError (verified
      the id flows end-to-end). CLI path (`run_agent`) doesn't seed it — traces show `<OceanState>`.
- [x] `core/` move done (2026-07-03): `config/exceptions/logging_config/llm/prompts` +
      `commons` → `src/core/`; ~22 import rewrites to `core.*`; `pyproject` include updated;
      verified (compileall + all-layer import smoke + behavioral smokes pass).
- [x] Committed the restructure + Arc A (2026-07-06): `src/agents/` → `core/` + `app/`,
      the injection spine, and the two fixes below.
- [x] Arc A done + verified live via `debug_driver` (2026-07-06): `build_context()` →
      `AppContext(settings, graph, deps)`; LLM registry (A1) + prompt registry (A2, template
      at `core/prompts/templates/oceans_agent/v1/`) wired and consumed in `graph.py`.
- [x] Two restructure regressions found + fixed (2026-07-06): (1) `bind_tools` was dropped
      when the LLM moved to the registry — restored in `make_agent_node`, so tools fire again;
      (2) `temperature` deprecated@opus-4-8 came back via the registry's hardcoded
      `temperature=0` — now config-driven (`LLMModel.temperature`, opus-4-8 sets `None`).
- [ ] Cosmetic (X1): world-simulator docstring headers still in `core/config.py`,
      `core/agents/*`, `core/llm/llm_registry.py` (stale `from config import
      build_llm_registry` example, etc.).

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
