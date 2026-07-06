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

### B1 — conversation state (checkpointer) — ✅ DONE 2026-07-06

MemorySaver; wired + verified live (turn 2 recalled "556" + range, zero tool calls). B1.5 upgrades
the saver to Postgres for durability.

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

### B1.5 — durable checkpointer (Postgres) — ✅ DONE 2026-07-06

Harvest `journal_agent/stores/checkpointer.py` — it already runs `AsyncPostgresSaver` with a
lifecycle-managed pool. NOT the reflection code.

- **B1.5.1 — Decide: stay on MemorySaver, or move to Postgres.** `[YOU · tier-3 · dep: B1.4]` ✅
  **DECIDED: move to `AsyncPostgresSaver`.** MemorySaver dies on restart; a conversation should
  outlive a process bounce. The swap is essentially one line at the composition root and the
  saver API is identical, so there's little cost to taking durability now.
  **Done when:** decision recorded here. ✅

- **B1.5.2 — Wire AsyncPostgresSaver.** `[ME · tier-1 · dep: B1.5.1]` ✅
  `build_context` is now an `@asynccontextmanager` that yields the saver from a nested async CM
  (`stores/checkpointer.make_postgres_checkpointer` → `from_conn_string(url)` + `.setup()`), torn
  down at block exit. BOTH entry points hold it open correctly: `debug_driver.main` and the server
  `lifespan` (`ocean_runner.py`) each wrap it in `async with build_context() as service:` so the
  pool closes at shutdown. `thread_id` unchanged (= `session_id`).
  **DEVIATION from the original ticket:** it does NOT "reuse the existing pg (`stores/postgres`)".
  The checkpointer opens its OWN async pool — correctly, because LangGraph's async saver needs
  async psycopg while `PgGateway` is sync. Two pools by design, not oversight.
  **Done when:** the compiled graph carries a Postgres checkpointer; tables auto-create. ✅

- **B1.5.3 — Keep OceanState thin (serde guard).** `[YOU · tier-2 · dep: B1.5.2]` ✅
  Journal had to pre-register ~15 domain types (`_ALLOWED_MSGPACK_MODULES`) or the checkpointer
  dropped nested Pydantic on roundtrip. Oceans' state is `messages` + scalars → default JsonPlus
  serde works with ZERO registration (`_ALLOWED_MSGPACK_MODULES = []`). Rule: don't put rich domain
  objects in `OceanState`; if you must, register them like journal did. This is why thin state is a
  feature.
  **Done when:** a two-turn conversation survives a full process restart. ✅ **PROVEN 2026-07-06**
  via a two-process test (fixed `session_id`): process A ran turn 1 (photo → id 556) and exited;
  process B (fresh) ran turn 2 and recalled "556" + the exact core range with **zero tool calls** —
  only possible if Postgres persisted turn 1 across the restart. MemorySaver would have returned empty.

### B2 — context assembly (rework journal's `ContextBuilder`, made generic)

Harvest `journal_agent/configure/context_builder.py` — NOT the reflection graph. It already does
**budget + priority-prune** (retrieved → recent → session, to a token budget). Two facts banked
from the comparison: (a) journal trims at **CALL TIME** — rebuilds the LLM input each turn, never
mutates persisted state — which SIDESTEPS the `RemoveMessage` / tool_use↔tool_result pairing
problem that durable pruning of `messages` would hit; (b) journal's builder is fed by a classifier
node emitting a per-turn `ContextSpecification` — oceans' ReAct loop has no such node, so decide
what produces the "spec" here.

- **B2.0 — Decide the assembly shape (inline here, NOT a separate doc).** `[YOU · tier-3 · dep: B1 done]`
  Your design frontier — do NOT let me pre-break it. Decide: what a generic assembler owns (budget +
  priority prune à la `ContextBuilder`), ephemeral (call-time) vs durable pruning, where it runs
  (inside `make_agent_node`? a pre-model hook?), and what supplies the per-turn spec absent a
  classifier node. Keep the ContextBuilder token-budget machinery; strip its journal-specific inputs.
  **Done when:** the shape is written into THIS section (a short B2 subsection); THEN we ticket the build.

  ---
  **B2.0 — design-space notes (captured 2026-07-06; decisions still OPEN — work through tomorrow).**
  Context: `src/core/context_builder.py` was brought over from journal_agent and is half-annotated;
  it's fighting us because journal was single-shot and oceans is a ReAct loop. The framing below is
  the pre-work; the 5 decisions at the bottom are what's actually mine to make.

  - **THE REFRAME (changes everything):** journal's `ContextBuilder` runs ONCE per turn (classifier →
    spec → assemble `[system, recent, session]` → one LLM call). Oceans is a **ReAct loop**: the agent
    node calls the LLM repeatedly within ONE turn, and `messages` GROWS mid-turn (each tool call appends
    `tool_use` + `tool_result`). So (a) context assembly runs before EVERY LLM call, inside
    `make_agent_node` — not once at the top of the turn; (b) fat tool outputs (e.g. `vessel_traffic`'s
    huge JSON) are the main budget threat, not old conversation. Journal's recent-vs-session vocabulary
    answers a question we don't have.

  - **Tokens — measure before AND after, different jobs:** estimate BEFORE each call (cheap chars/4
    heuristic) to DECIDE trimming; read ACTUAL counts AFTER from `usage` metadata (`core/llm/
    token_callback.py`) to observe/calibrate. GOTCHA: `tiktoken` is OpenAI's tokenizer — it doesn't know
    Claude model names, so the tiktoken branch in the brought-over file silently falls to the estimate
    every time (dead weight). Real Claude counts = Anthropic `count_tokens` API, else own the heuristic.

  - **Kill the hardcoding via "sections with a policy" (generalizes my two-types instinct):** a context
    section = `{ content, priority, policy, pinned? }`. `policy` = how it shrinks (`drop-oldest` |
    `truncate` | `keep-first+last` | `summarize`). "tool data" vs "messages" aren't the taxonomy — they're
    two section INSTANCES wanting different policies (fat tool result → `truncate`; old turns →
    `drop-oldest`). Journal hardcodes 3 sections + 1 policy; the seam is: sections are data, policies pluggable.

  - **Maturity ladder + where the over-engineering line is FOR THIS PROJECT:**
    L0 do nothing (let it grow into Opus 200k) · L1 sliding window (last N, clean tool pairs) ·
    **L2 = section budget + priority + per-section policy, computed at call time ← our honest target** ·
    L3 summarization/compaction (Claude Code `/compact`, LangGraph `SummarizationNode`) ·
    L4 semantic/retrieval memory across sessions (Letta/MemGPT, Zep). L3 is where over-engineering starts
    for now — but design the L2 seam so `summarize` is just an unimplemented `policy` value. L4 waits on B3.

  - **THE LANDMINE (also simplifies the design):** in a ReAct loop you CANNOT pop from the end of persisted
    `messages` — you'll orphan a `tool_use` from its `tool_result` → Anthropic 400. Escape hatch (doc fact
    *a*): trim at CALL TIME, never mutate persisted state. So make the builder a PURE FUNCTION:
    `persisted OceanState → message list for THIS call` (no write-back). Checkpointer keeps full history
    durable; pruning is ephemeral/per-call. Kills the orphaned-pair bug class AND is the reusable pattern.
    Durable compaction (`RemoveMessage`) is an L3 concern to defer.

  - **THE 5 OPEN DECISIONS (mine; Claude to rail + push back tomorrow):**
    1. Ephemeral vs durable pruning. *(Claude recommends ephemeral/call-time, strongly — the landmine.)*
    2. Adopt "sections with a policy"? v1 policy set? *(Claude: `truncate` tool results + `drop-oldest` turns
       is enough to be interesting.)*
    3. Where it runs — inside `make_agent_node` before `llm.invoke`, as a pre-model hook. Confirm.
    4. What supplies the "spec" — no classifier node here. v1 = static config (budget + per-section policy);
       `/skill`-selected spec is the B3 upgrade. *(Claude: yes, don't build the classifier.)*
    5. Real budget number for Opus — NOT the inherited `max_tokens=8000` (journal-era small-model ceiling).
    Decision #1 comes first; everything hangs off it.

- **B2.1+ — TBD after B2.0.** Ticketed once the shape is decided, Arc-A style.

### B3 — `/skills`: context commands (the menu)

A named, reusable bundle that declares **what to pull into context for a class of request** (tools +
data + instructions) — the `/skills`-shaped front end to B2's assembly (memory:
`oceans-direction-context-commands`). Journal does a lightweight version: `user_command` dispatch
(`/reflect`, `/recall`, `/save`) each pairing to a per-command `ContextSpecification`.

- **B3.0 — Discuss/design the skill shape.** `[YOU · tier-3 · dep: B2.0]`
  Do NOT pre-ticket. Decide: what a skill IS (a named context bundle, NOT a procedure to run), how
  it's selected (explicit `/command` vs inferred from the question), how it composes with B2 assembly
  and the ReAct tool set. Journal's `/command` dispatcher is the concrete precedent to borrow from.
  **Done when:** the shape is decided + noted here.

---

## Arc C — orchestration (the agent calls workers)

Turn the single ReAct agent into an **orchestrator** that delegates to worker agents/graphs exposed
to the LLM **as tools** (a worker = a tool whose body is itself a sub-graph). Lets the top agent
decompose a hard question across specialists without one monolithic prompt/tool set.

- **C1.0 — Discuss/design worker-as-tool.** `[YOU · tier-3 · dep: none; saner after B2]`
  Do NOT pre-ticket. Decide: the binding seam (same `bind_tools`; the tool body does
  `subgraph.ainvoke(...)`), shared vs isolated state/checkpointer per worker, how a worker's result
  re-enters the orchestrator's message stream, whether workers stream tokens. Contrast journal, which
  invokes sub-graphs INSIDE a node (`reflection_graph.ainvoke` from `make_reflect_node`) — a scripted
  call, NOT an LLM-chosen tool; the agentic version lets the model decide when to call a worker.
  **Done when:** the pattern is decided + noted here; THEN ticket the build (tier-1 plumbing = mine).

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

**B1 + B1.5 (checkpointer, now DURABLE) DONE + verified (2026-07-06).** `build_context`
builds the saver → `build_sandbox_graph(deps, saver)` compiles with `checkpointer=` (B1.2);
`ocean_runner`/`ask` passes `thread_id=session_id` (B1.3). B1.4 proven live (two turns, one
process, turn 2 recalled "556" with zero tool calls). B1.5 upgraded MemorySaver →
`AsyncPostgresSaver` (async CM, own pool, teardown on shutdown in BOTH entry points) and proved
DURABILITY across a real process restart (two-process test: process B recalled turn 1 from
Postgres after process A exited). Server `lifespan` bug fixed along the way — it was assigning the
un-entered context manager to `runner.service`; now `async with build_context() as service:`.

Looking forward (added 2026-07-06 from the journal_agent infra comparison — harvest journal's
plumbing, NOT its reflection/summarization domain code):

- ~~**B1.5** — durable **Postgres checkpointer**~~ ✅ DONE 2026-07-06 (see B1.5 section).
- **B2** — rework journal's **`ContextBuilder`** into a generic context assembler (budget + priority
  prune). Your design frontier (B2.0), inline in this doc — no separate note.
- **B3** — **`/skills`** context commands: the menu of named context bundles (the front end to B2).
- **Arc C** — the agent becomes an **orchestrator** that calls worker sub-graphs as tools.

Unblocked frontier-discussion tickets (all yours, all "decide the shape first, don't let me
pre-ticket"): **B2.0** (assembly shape) · **C1.0** (worker-as-tool). B3.0 waits on B2.0. Feeds B2: durable pruning of `messages` must cut on clean turn boundaries (Anthropic
tool_use/tool_result pairing) and use `RemoveMessage`/`REMOVE_ALL_MESSAGES` — but journal's
call-time trimming sidesteps that entirely, which is the strong argument for starting B2 ephemeral.
