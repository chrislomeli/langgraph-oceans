# Public-repo gap analysis — would strangers clone this?

> Companion to `reuse-gap-analysis.md` (2026-07-24), which asked the INTERNAL question:
> "can I lift this into my next project?" This doc asks the harder PUBLIC question:
> "should this be its own repo that people who don't know me clone and use?"
> Written 2026-07-25, from a fresh read of every file under `src/core/`.

## The public bar is a different test

The internal litmus (import purity, no side effects, inverted config, declared
surface, honest docs, portable tests) measures whether code can *move*. The public
bar measures whether code can *compete*. A module clears it only if all four hold:

1. **A stranger gets value in 10 minutes** — README, quickstart, `pip install`, one
   working example, without any oceans context.
2. **It beats the incumbent** — there is a concrete reason to use this instead of
   what people already reach for (or nothing they reach for exists).
3. **Evidence** — tests a stranger can run, and some sign it has survived real use
   beyond one author's one app.
4. **A maintenance story** — nothing inside that rots on its own (hardcoded model
   catalogs, price tables) unless you intend to keep tending it publicly.

Being well-crafted is necessary but nowhere near sufficient. Most good internal
code fails checks 2–3, and that is not an insult — it means the code is scaffolding,
not product. The verdicts below try hard not to confuse the two.

## Scorecard

| Module | Internal verdict (prior doc) | Public verdict | One-line reason |
| --- | --- | --- | --- |
| `core/context` | ✅ package-ready | 🟡 **the only candidate** — not yet, but the idea earns the work | differentiated design vs. langmem/SummarizationNode; missing proof + consumer docs |
| `core/prompts` | 🟡 two small cuts | ❌ keep internal | ~160 lines in a crowded space; an afternoon's work to rewrite |
| `core/llm` | 🟡 small gaps | ❌ keep internal | thin layer over LangChain ctors + a price/model table that rots |
| `core/agents` | 🟡 umbrella | ❌ keep internal | standard-issue plumbing; and it leaks a domain import (new finding) |
| `core/config` | ❌ | ❌ (trivially) | oceans settings monolith; already ruled out |
| `exceptions`, `logging_config` | not deep-read | ❌ (trivially) | too small to be anything but internal glue |

**Honest headline:** of six modules, exactly one has a public-repo case, and even it
isn't ready today. The rest are competent internal scaffolding whose value to you is
real and whose value as standalone repos is ~zero — publishing them separately would
neither help others nor showcase you. That's the normal ratio; believing otherwise
is the failure mode this doc exists to prevent.

---

## `core/context` — 🟡 the one real candidate

### Why it clears bar #2 (differentiation) — genuinely

This is not a me-too summarizer. Read cold, the design has ideas the incumbents
don't have, and they compose:

- **Never mutate the log.** `messages` is the sole ground truth and grows forever;
  everything else is derived shadow state (`message_summaries`, `tool_calls`,
  bookmark) kept only because it's expensive to recompute. LangGraph's own
  `SummarizationNode` / `RemoveMessage` idiom *rewrites* the message channel —
  lossy and checkpoint-hostile. This design is lossless by construction.
- **Two streams, two treatments.** Prose is summarized; tool traffic is *briefed*
  into a leading manifest ("have I called this already?"). Nobody's off-the-shelf
  compressor makes that distinction, and for ReAct loops it's the right one — tool
  dedup is exactly what agents forget.
- **Ephemeral fitting.** The hard ceiling sheds from the outgoing copy only
  (tail → summaries → briefs, current turn non-negotiable); dropped items return
  next turn because state was never touched. Clean separation of durable
  compression from per-call fitting.
- **Checkpoint-resume-safe by id-bookmark**, failing loud on a missing id.
- **Prompt-cache-aware placement** (manifest as a leading framed turn, never in the
  system prompt) — a real-world consideration most published examples ignore.

Craft matches the ideas: pure functions, protocol-based injection done correctly
(implementers never import the protocol file), curated `__init__`, frozen value
objects, fail-loud policy resolution, and the only tests in `core/`. This is the
best-written code in the repo by a distance.

### Why it does NOT clear the bar today

- **No evidence (bar #3, the big one).** It has run inside one app, by one author,
  for about two weeks, with — per the module's own comments — the retention seam
  unwired and `view_token_ceiling` defaulting to off. A context-compression library
  earns trust through miles, and it has none yet. Nothing shameful; just true.
- **Test coverage is one file.** `tests/core/context/test_summarization.py`
  (494 lines) is a real start, but a public library needs the fitter cascade,
  pair-safety, ordinal monotonicity, bookmark-resume, and the overflow error path
  each pinned separately — those invariants ARE the product.
- **The docs are for us, not for strangers (bar #1).** The docstrings are excellent
  *internal* docs — and that's the problem. They reference "Option B", "the design
  discussion", "see the ordinal bug", "case-4 last resort", `docs/design/
  Summarization.md`, "open question #4". A stranger hits those in the first minute
  and correctly concludes this is someone's project internals with the door left
  open. There is no README, no quickstart, no packaging metadata.
- **No positioning.** The README that doesn't exist must answer, in its first
  screen: "why this and not `langmem` / `SummarizationNode` / `trim_messages`?"
  The answer exists (above) — but nobody will derive it from source.
- **Sync summarizer in an async world.** `LLMSummarizer.summarize` calls
  `invoke()`; the host graph is async. Public users on async graphs will hit this
  in week one.
- **Small teach:** `HeuristicTokenCounter` as a default is fine, but a public repo
  must state its error bars — a fitter with a hard "ceiling" enforced by a ~25%-off
  estimator is a ceiling in name only unless documented as such.

### The honest disposition

Don't repo it now, and don't discard the ambition. The cheap, high-value move:
keep hardening it *inside* oceans (async summarizer, fitter tests, wire retention),
and let the **maintainer-agent project be consumer #2** — a second real consumer is
worth more than any refactor for proving the API is genuinely generic. If it
survives that with its interface intact, the extraction is mechanical and the
README practically writes itself from `Summarization.md`. If instead you want the
public credit *now*, the honest format is a design write-up (blog post / repo of
one annotated module) — "here's a context layer and why", not "here's a library you
should depend on". A library promise you can't back with miles hurts more than no
repo.

---

## `core/prompts` — ❌ as a repo; fine as internal kit

What it is: a ~160-line Jinja2 loader with versioned template dirs,
`manifest.yaml` required-vars, a `schema` filter for Pydantic models, fail-loud
everything. The craft is fine — `StrictUndefined`, no silent fallbacks, correct
dependency direction.

Why it fails the public bar: **check 2, decisively.** Versioned-prompt management
is a crowded, commoditized space (LangSmith/hub, PromptLayer, banks, a hundred
gists), and the floor alternative — "a folder of Jinja files and twenty lines of
loader" — is something every capable consumer can write faster than they can
evaluate your repo. There is no idea here a stranger can't reproduce from the
one-sentence description. The `schema` filter is the only distinctive touch, and
it's a feature, not a product.

Secondary gaps (would matter only if you overruled the verdict): domain templates
ship inside the package (`templates/oceans_agent/`), it raises `core.exceptions.
PromptError`, lexicographic version sort is a footgun past `v9`, no tests.

Disposition: keep it, use it in the next project, never apologize for it — and
don't repo it. If you want public mileage from this code, it's one section in a
"how I structure agent projects" write-up.

## `core/llm` — ❌ as a repo; useful internally with caveats

What it is: role→model indirection with lazy factories + `warmup()`, per-role
token/cost accounting, and provider-aware `make_system_message` cache markup.
Those three are genuinely nice *features* — the lazy-with-optional-warmup
trade-off is correctly reasoned and correctly documented.

Why it fails the public bar:

- **Check 2:** it's a thin dispatch layer over LangChain's own constructors, and
  LangChain already ships `init_chat_model(provider:model)`; heavier users reach
  for litellm or a gateway. The role-indirection idea is sound but is ~50 lines of
  anyone's composition root.
- **Check 4 (maintenance rot), structurally:** the module embeds a hardcoded model
  catalog with list prices as code. In a private repo that's a pragmatic cache; in
  a public repo it's a treadmill — stale the month after publishing, and wrong
  prices in a *cost-reporting* library are worse than no prices.
- **Check 5-style honesty failure, worse than the prior doc recorded:** the module
  docstring of `llm_registry.py` is a verbatim copy of `config.py`'s — it
  literally opens "world-simiulator.config / Centralised settings…" and its usage
  example imports `LLM_ROLE_CONFIG`, a symbol that no longer exists anywhere. The
  file's front door describes a different file in a dead codebase. Fix this
  regardless of any repo decision.
- No tests; fat `Settings` coupling (prior doc, still true).

Disposition: keep internal. The extractable public-worthy *nugget* is
`make_system_message`'s provider-aware cache markup + the token/cost callback —
blog-post material, not a package.

## `core/agents` — ❌ as a repo; and one new finding

What it is: `node_executor` (sync/async decorator: timing, error→`NodeError`
state, session tracing), `node_metrics`, `TracedState`, `routing` (unused),
`dependencies.py`.

Why it fails the public bar: **this is standard-issue plumbing.** Every LangGraph
shop writes a node decorator like this; the observability it provides is the
territory of LangSmith/OpenTelemetry callbacks that users already have. There is
no distinctive idea to anchor a repo on. Internals confirm "internal-grade":
duplicated sync/async wrapper bodies, a module-global `metrics` singleton, zero
tests, and the world-simulator docstring headers (X1) still in place.

**New finding — the prior doc's "check 1 clean sweep" was wrong.**
`dependencies.py:20` does `from stores.postgres import PgGateway`. `stores/` is
domain code, so `core/` is NOT import-pure today; the 2026-07-24 sweep only
grepped for `agents.`/`app.` and missed `stores.`. Correct fix is the same as the
prior doc's direction for this file (an `AgentDependencies` container belongs to
the app, or the store becomes a protocol) — but the scorecard line "check 1 is a
clean sweep" in `reuse-gap-analysis.md` should be amended.

Disposition: keep internal; fix the `stores` leak because it breaks even the
*internal* litmus, not for publishing's sake.

## `core/config`, `exceptions`, `logging_config` — ❌ trivially

`config.py` already failed the internal bar (oceans monolith: whale-fluke
`image_root`, `embedder_ver`, hardcoded `LANGSMITH_PROJECT`); the public bar is
strictly higher. The other two are a handful of lines of glue. No one clones a
repo for an exceptions file. Nothing more to say.

---

## What to actually do with this

1. **Amend `reuse-gap-analysis.md`:** check 1 is not a clean sweep —
   `core/agents/dependencies.py` imports `stores.postgres` (domain). Re-run the
   sweep including `stores.`, `models.`, `rag.`.
2. **Fix the two honesty failures now, cheap, repo-or-not:** llm_registry's
   copied-from-config docstring + dead `LLM_ROLE_CONFIG` example; the X1
   world-simulator headers.
3. **Treat `core/context` as the one public bet, on a proof timeline, not a
   publish timeline:** async summarizer, invariant tests (fitter, pair-safety,
   ordinals, resume), wire retention — then let the maintainer agent be consumer
   #2. Extraction decision after that, with evidence in hand.
4. **Re-aim the rest at the write-up, not the registry.** The prompts/llm/agents
   modules are portfolio material as *prose* ("how I structure agent projects")
   and liabilities as repos. Publishing five thin repos would dilute the one
   strong story you have.

## Seed-corpus rules harvested (maintainer-agent recursion)

- "Internal-reusable and public-worthy are different bars: movability vs.
  competitiveness. Test them separately."
- "A public library needs a maintenance story: hardcoded catalogs/prices are
  caches internally and treadmills publicly." (llm)
- "Docstrings that cite internal design discussions ('Option B', 'the ordinal
  bug') mark code as internal — great for the team, disqualifying for a public
  README." (context)
- "An import-purity sweep must enumerate ALL domain roots, not the two obvious
  ones." (agents/dependencies — the missed `stores.` import)
- "A copied module docstring is worse than doc-rot: the file's front door
  describes a different file." (llm)
- "N thin repos < 1 proven repo + 1 good write-up." (portfolio strategy)