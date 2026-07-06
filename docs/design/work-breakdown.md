# Work breakdown — injection spine + context layer

*Drafted 2026-07-04. Dev-ticket granularity (not use cases). The point: "what's next" is
always the top unblocked ticket. Owners follow the tier split (memory: agentic-division-of-
labor). Parent design: `app-context-design.md`. This supersedes that note's §Build order.*

**Legend** — `[ME]` tier-1 plumbing (I write, you review) · `[YOU]` tier-2 agentic core /
tier-3 design (you write/decide, I rail + review) · `[BOTH]` a verification run.
Each ticket names its **dep** and its **done-when** (usually observable in `debug_runner`).

---

## Arc A — the injection spine (mostly plumbing; behavior-preserving)

Goal: kill the `GRAPH = build_graph()` global; everything the runner needs arrives via a
context object built once at startup. No behavior change until A1.

### A0 — composition root

- **A0.1 — Decide the context object shape.** `[YOU · tier-3 · dep: none]`
  Extend `AgentDependencies` with a `graph` field, or wrap it: `AppContext(deps, graph)`.
  *Recommend:* an `AppContext` wrapper — keeps the ported `AgentDependencies` a generic
  ingredient-holder; the compiled graph is *built from* those ingredients.
  **Done when:** the shape is written into `app-context-design.md`.

- **A0.2 — Write `app/context.py::build_context()`.** `[YOU · tier-2 · dep: A0.1]`
  Assemble the context: `get_settings()`, `apply_langsmith()`, build the graph (the SAME
  way it's built today — don't touch `_llm` yet), return the context object. ~20 lines.
  Your DI rep; the design note is the guardrail.
  **Done when:** `build_context()` returns a context with a compiled `graph`, no import-time work.

- **A0.3 — `ocean_runner(request, ctx)`.** `[ME · tier-1 · dep: A0.2]`
  Change the signature; read `ctx.graph`, not the module global.
  **Done when:** runner no longer references a global `GRAPH`.

- **A0.4 — Server wiring.** `[ME · tier-1 · dep: A0.3]`
  In `chat_app`: `lifespan` builds `ctx` into a holder; closure `runner(req)` calls
  `ocean_runner(req, ctx)`; pass both to `create_chat_app`. Delete `GRAPH = build_graph()`.
  **Done when:** no module-level graph build; uvicorn target still imports.

- **A0.5 — Debug wiring.** `[ME · tier-1 · dep: A0.3]`
  `debug_runner.main`: `ctx = build_context()`; pass to `ocean_runner`.
  **Done when:** `debug_runner` builds its own ctx, no server.

- **A0.6 — CHECKPOINT: prove injection.** `[BOTH · dep: A0.4, A0.5]`
  Run `debug_runner` (needs `.env`/API). Same trajectory as today, now dep-injected.
  **Done when:** one streamed turn completes and `grep GRAPH src` shows no global.

### A1 — wire the LLM registry

Verdict: the registry (`core/llm/llm_registry.py`) is already seasoned-grade. **Wire, don't rebuild.**

- **A1.1 — Add `claude-opus-4-8` to the model catalog.** `[ME · tier-1 · dep: none]`
  `models` in `llm_registry.py` tops out at `claude-opus-4-7`; add/replace with `-4-8`.
  **Done when:** an OPUS-4.8 `LLMModel` entry exists with prices.

- **A1.2 — Define ocean roles.** `[YOU · tier-3 · dep: none]`
  Replace world-sim `LLM_ROLE_CONFIG` (classifier/logistics/code_intel) with ocean roles —
  at minimum `"ocean_agent": OPUS`. Decide if a second cheap role is worth reserving now
  (future router/judge) or added when needed.
  **Done when:** `LLM_ROLE_CONFIG` names only roles the ocean code actually requests.

- **A1.3 — Decide the tool-binding seam.** `[YOU · tier-3 · dep: none]`
  Registry returns a generic model; where does `.bind_tools(TOOLS)` live?
  *Recommend:* in `build_graph`/agent, not the registry (binding is agent-specific).
  **Done when:** decision recorded.

- **A1.4 — Swap `build_graph` onto the registry.** `[YOU · tier-2 · dep: A1.1–A1.3, A0.2]`
  `build_context` builds the registry (`build_llm_registry(...)`) into the context;
  `build_graph(llm_registry)` does `registry.get("ocean_agent").bind_tools(TOOLS)`.
  Delete `_llm()` and the `MODEL` constant.
  **Done when:** graph uses the registry; `_llm`/`MODEL` gone.

- **A1.5 — VERIFY.** `[BOTH · dep: A1.4]`
  `debug_runner` run: same answer via the registry; `registry.usage_report()` shows tokens.
  **Done when:** a turn runs and reports token usage per role.

### A2 — wire the prompt registry

- **A2.0 — Read + assess `core/prompts/registry.py`.** `[ME · tier-1 · dep: none]`
  Report what it does, how prompts are stored/loaded, whether it needs improvement. (Blocks
  the rest of A2 — can't wire what we haven't assessed.)
  **Done when:** a short assessment is posted here.

- **A2.1 — Decide prompt storage.** `[YOU · tier-3 · dep: A2.0]`
  Where does `SYSTEM_PROMPT` live — a template file the registry loads, or inline? (Tool ads
  are a later, separate question.)
  **Done when:** decision recorded.

- **A2.2 — Move `SYSTEM_PROMPT` into the registry.** `[YOU · tier-2 · dep: A2.1, A0.2]`
  `build_context` builds `prompt_registry`; the agent fetches the system message via it.
  *Bonus:* use `llm_registry.make_system_message(role, text)` so Anthropic `cache_control`
  is applied → system prompt is prompt-cached (~10% cost on hits). A seasoned detail.
  **Done when:** `SYSTEM_PROMPT` no longer a bare module string; cache markup present.

- **A2.3 — VERIFY.** `[BOTH · dep: A2.2]`
  `debug_runner`: same behavior; confirm the system message carries `cache_control`.
  **Done when:** a turn runs and the cached-prefix is confirmed (usage shows cache read).

---

## Arc B — the context layer (the portfolio work; yours)

### B1 — conversation state (checkpointer)

- **B1.1 — Decide the saver.** `[YOU · tier-3 · dep: none]`
  `MemorySaver` (in-proc, dies on restart) vs `SqliteSaver`. *Recommend:* MemorySaver to start.
  **Done when:** decision recorded.

- **B1.2 — Attach the checkpointer.** `[ME · tier-1 · dep: B1.1, A0.2]`
  `build_context`/`build_graph` compiles with `checkpointer=`.
  **Done when:** the compiled graph carries a checkpointer.

- **B1.3 — Thread `thread_id`.** `[ME · tier-1 · dep: B1.2]`
  `ocean_runner` passes `config={"configurable": {"thread_id": request.session_id}}`.
  **Done when:** invoke carries the thread config.

- **B1.4 — VERIFY multi-turn.** `[YOU · dep: B1.3]`
  In `debug_runner`, call `ocean_runner` twice with the SAME `session_id`, different
  questions; turn 2 references turn 1 (e.g. "what did I just ask?").
  **Done when:** the terminal shows memory carrying across two turns.

### B2 — context assembly (the `/skills`-shaped per-request bundle)

- **B2.0 — Design note first.** `[YOU · tier-3 · dep: B1 done]`
  This is your design frontier — do NOT let me pre-break it into tickets (that's me doing the
  thinking). Write the mid-level note: what a named context bundle *is* (tools + data +
  instructions to load for a class of question), how it's selected, how provenance is tracked.
  I advise both sides; you decide the shape.
  **Done when:** `docs/design/context-assembly-design.md` exists; THEN we ticket it like Arc A.

---

## Background / cleanup (tier-1, do anytime)

- **X1 — Cosmetic docstrings.** `[ME]` Strip `world-simulator` headers from `core/config.py`,
  `core/agents/*`, `core/llm/llm_registry.py` (incl. the stale `from config import
  build_llm_registry` example).
- **X2 — Commit the restructure.** `[BOTH]` The `app/`, `commons`-weed, `OceanState`, and
  `core/` moves are all uncommitted. Commit at a green checkpoint (after A0.6 is natural).

---

## Start here

**Arc A is DONE and verified end-to-end (2026-07-06).** `build_context() → AppContext(settings,
graph, deps)`; LLM registry (A1) + prompt registry (A2) wired and consumed; `debug_driver` runs
the full F5 chain, every node `status: ok`. Two restructure regressions were caught + fixed on
the way: `bind_tools` (dropped when the LLM moved to the registry) and `temperature`
deprecated@opus-4-8 (registry's hardcoded `temperature=0` — now config-driven via
`LLMModel.temperature`, opus-4-8 sets `None`). X2 (commit the restructure) is done.

Top unblocked ticket: **B1.1** — you decide the checkpointer saver (under discussion, not yet
decided). Then B1.2–B1.4 (attach saver → thread `thread_id` → multi-turn verify) → B2.0 (the
context-assembly design note, your frontier).
