# Design note — the app context & composition root

*Status: drafted 2026-07-03, tier-3 decisions open. Mid-level build spec for wiring
bucket-3 infrastructure into the two entry points (`chat_app.py`, `debug_runner.py`).
Parent: none — this is the runtime substrate the agent graph runs inside. Decisions
marked **DECIDE** are yours (tier 3).*

## What it is

The single place that **builds shared infrastructure once** and **hands it to the runner**,
so nothing self-instantiates at import time and the front end owns none of it.

The driving distinction — three kinds of state, three homes:

| Bucket | What | Scope | Home |
|--------|------|-------|------|
| 1. Conversation | message history | per-session (keyed) | inside the checkpointer |
| 2. Request identity | `session_id`/`thread_id`, flags | per-request | on `TurnRequest` |
| 3. Infrastructure | graph, LLM registry, prompts, stores, settings | **app-wide, one each** | the app context |

This note is about **bucket 3**: app-scoped, built at startup, session-keyed *data* held
*inside* the components — never a component per session.

## The core rule

- `ocean_runner` is **stateless across turns**. It is a pure function of `(request, ctx)`:
  one question in, frames out, remembers nothing between calls. Cross-call memory lives in
  the checkpointer (bucket 1); shared machinery lives in `ctx` (bucket 3).
- The composition root **constructs** bucket 3 and **hands it in**. The runner never reaches
  outward for a global. This kills the current `GRAPH = build_graph()` import-time global in
  `chat_app.py` — the thing we explicitly don't want.

## Where things live (layout) — DECIDED 2026-07-03

The platform substrate does **not** belong under `agents/commons` — nesting the general
foundation under one feature is a scope inversion, and `commons` is a junk-drawer name that
attracts anything. Promote it to a top-level `core/` package, governed by one rule that
decides membership for every file:

> **`core/` imports nothing domain-specific.** Dependencies point one way:
> **domain → core, never back.** If a file wants to import the ocean agent / tools / models,
> it isn't platform — it's domain, and it stays out of `core/`.

Three tiers:

```
src/
  core/                    # PLATFORM — domain-agnostic; imports nothing ocean
    config.py              # ← from root
    exceptions.py          # ← from root
    logging_config.py      # ← from root
    llm/                   # ← moves in whole (LLMRegistry, token_callback)
    prompts/               # ← moves in whole (PromptRegistry)
    agents/                # generic agent-graph framework — NOT the ocean agent
      node_executor.py     # ← from agents/commons
      node_metrics.py
      node_types.py
      routing.py
      state_types.py
      dependencies.py      # AgentDependencies (the container shape)

  agents/sandbox_agent/    # DOMAIN — the ocean agent (depends on core)
  tools/  models/  stores/  rag/     # DOMAIN capabilities
  evals/  data/  training/           # DOMAIN offline scripts

  app/                     # SHELL — the entry layer; knows BOTH, wires them together
    context.py             # build_context() — the composition root
    chat_app.py            # ← from root
    debug_runner.py        # ← from root
```

The old `agents/commons` splits: the *generic agent-graph scaffolding* (node executor,
metrics, routing, state, the `AgentDependencies` shape) is real platform → `core/agents/`;
the *ocean agent itself* stays out in `agents/sandbox_agent/`. The name `commons` disappears.

**The composition root is deliberately NOT in `core/`.** `build_context()` is the one place
allowed to know both worlds (it picks the ocean graph + ocean roles and wires them into the
platform container), so by definition it violates the one-way rule. It lives in the `app/`
shell with the entry points — which also satisfies "no logic in `src` root": root becomes
just packages.

Naming: `core/` chosen over `platform/` (both fine), avoiding `framework/` (overclaims) and
`commons`/`utils`/`shared` (junk-drawer). The *dependency rule*, not the name, does the work.

## The container (hold-it)

Adopt the already-ported `AgentDependencies` (moving to `core/agents/dependencies.py` per
the layout above) — it is exactly the app-context shape: a dumb holder with `llm_registry`,
`prompt_registry`, `store`. No behavior; if a method wants to *do work*, that work belongs in the graph or the
runner, not here. The container's only jobs are **hold** and (via a factory) **build**.

## The composition root (build-it)

One function whose whole job is to assemble the container:

```
build_context() -> AgentDependencies
    settings   = get_settings()
    llm        = build_llm_registry(settings, models, LLM_ROLE_CONFIG)   # ported
    prompts    = PromptRegistry(...)
    graph      = build_graph(llm_registry=llm, checkpointer=<see DECIDE>)
    return AgentDependencies(llm_registry=llm, prompt_registry=prompts, ...)
```

It assembles the **ingredients** — the platform libs (`core/llm/`, `core/prompts/`) and the
domain graph (`agents/sandbox_agent/`). Living in `app/context.py`, it is the *one place*
allowed to know both worlds (see layout) and how the pieces fit together.

## The runner contract (from `agent_chat`, verified)

`agent_chat` pins the runner signature — `protocols.py`:

```python
class TurnRunner(Protocol):
    def __call__(self, request: TurnRequest) -> AsyncIterator[RunnerFrame]: ...
```

Three facts that drive the wiring:

1. **One positional arg** (`request`). The library calls `runner(request)` and nothing else
   (`transport.py:55`). It will never pass `ctx` — so injection **must** be a closure. Not a
   style choice; forced by the protocol.
2. **Returns async iterator of frames** — `ocean_runner` already matches.
3. **Must not yield `Done`/`Error`** — the transport appends exactly one terminal frame even
   if the runner raises (`transport.py:54-60`).

## The wiring (server path)

`create_chat_app(runner=..., lifespan=...)` takes both **at construction**, but `lifespan`
doesn't *run* until startup — so `ctx` doesn't exist when we call `create_chat_app`, and the
runner (one arg) can't see `app.state`. Reconcile with a **holder the runner closes over and
lifespan fills**:

```
holder = {}                              # empty at import — no heavy construction

async def lifespan(app):                 # runs at STARTUP, not import
    holder["ctx"] = build_context()
    yield
    # teardown (close pools) here later

async def runner(request):               # matches TurnRunner: one arg
    async for frame in ocean_runner(request, holder["ctx"]):
        yield frame

app = create_chat_app(runner=runner, lifespan=lifespan, title="Ocean Conservation Agent")
```

Sequence: construction wires `runner` + `lifespan` to a shared empty slot → startup runs
`lifespan`, fills it → first request reads the populated `ctx`. No import-time infra; the
runner receives its deps; the library only ever sees `runner(request)`.

```
uvicorn ─(startup)─► lifespan ─► build_context() ─► holder["ctx"]
browser ─(POST /chat/{id})─► transport ─► runner(request) ─► ocean_runner(request, ctx)
```

## The wiring (debug path)

No library scheduling startup, so no lifespan and no closure — build it by hand:

```
ctx = build_context()
async for frame in ocean_runner(request, ctx):
    ...
```

Same `build_context()`, same `ocean_runner`. That's why `debug_runner` stays a faithful
harness: both paths enter at the one seam with the same deps.

## Bucket 2 (session identity) — already handled by the library

The "client owns an id and resends it" model is implemented in `agent_chat`:

- `POST /sessions` mints a `uuid4` (`app.py:62`). Client calls once per conversation.
- `POST /chat/{session_id}` carries the id in the **path**; transport packs it into
  `TurnRequest.session_id` (`transport.py:89-95`).

Our only job is the one-line pass-through **inside** `ocean_runner`:

```python
config={"configurable": {"thread_id": request.session_id}}
```

`on_session_start` / `on_session_end` hooks (`app.py:53-54`) are available if we later want
to allocate/tear down per-conversation checkpointer state.

## DECIDE (tier 3 — your calls)

- **Adopt vs. slim.** ✅ DECIDED 2026-07-03: **adopt the ported `AgentDependencies` +
  `build_llm_registry` as-is, extend only if required.** Consequences to carry through the
  build:
  - `LLM_ROLE_CONFIG` carries world-simulator roles (classifier/logistics/code_intel — none
    ocean). **Replace with an ocean role(s)** — at minimum one role the agent graph requests
    (e.g. `"ocean_agent" → LLMLabel.OPUS`). Add `claude-opus-4-8` to the `models` catalog
    (current OPUS entry is `claude-opus-4-7`).
  - `graph.py::_llm` stops hand-building `ChatAnthropic`; instead `build_graph` takes the
    `llm_registry` and does `registry.get("ocean_agent").bind_tools(TOOLS)`.
  - "Extend if required" = add fields to `AgentDependencies` only when a real consumer needs
    them (e.g. a concrete `store` for bucket-1 memory). Don't pre-add speculative fields.
- **Checkpointer now or later.** Bucket-1 multi-turn memory needs `build_graph(checkpointer=…)`.
  `MemorySaver` (in-process) is a one-line start; `Sqlite/PostgresSaver` later, same API.
  DECIDE whether this note ships with memory or defers it to a follow-up.
- **Holder shape.** Bare `dict`/`nonlocal` vs. a tiny mutable object with a typed `ctx`
  attribute. Cosmetic; the mechanism is identical.

## Cleanups this surfaced (do when you touch these files)

- `config.py` docstring is **stale** — says `from config import build_llm_registry,
  LLM_ROLE_CONFIG, models`; those live in `llm/llm_registry.py` now.
- `agent_dependencies.py` header still says `world-simulator.agents.commons.deps`.
- Remove/env-gate the `[event]` debug print in `ocean_runner` (also fires under uvicorn).

## Build order

1. Write `build_context()` — assemble `AgentDependencies` from settings + registries + graph.
2. Change `ocean_runner` signature to `(request, ctx)`; read deps from `ctx`, not globals.
3. Server wiring: holder + `lifespan` + closure runner → `create_chat_app`. Delete the
   `GRAPH = build_graph()` module global.
4. Debug wiring: `ctx = build_context()` in `debug_runner.main`, pass to `ocean_runner`.
5. **Checkpoint:** run `debug_runner` — same trajectory as today, now dep-injected. That run
   proves the composition root without a browser.
6. Then (optional, per DECIDE): checkpointer + `thread_id` pass-through → call `ocean_runner`
   twice with one `session_id` in `debug_runner` to see multi-turn memory in the terminal.
