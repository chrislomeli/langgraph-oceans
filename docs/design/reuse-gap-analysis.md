# Reuse gap analysis — is `core/` actually packageable?

> Due diligence on the maintainer-agent reuse map (2026-07-24). The litmus: could each
> component live in its own repo as a package, with oceans as merely its first
> consumer? Passing the litmus matters; actually moving them is optional.
> Companion to `maintainer-agent-vision.md` (§ Reuse from oceans).

## The packageable litmus — 6 checks

1. **Import purity** — nothing in the package imports domain (`agents/`) or shell
   (`app/`) code. Mechanically checkable.
2. **No import-time side effects** — importing it doesn't read env vars, open files,
   or construct clients.
3. **Inverted configuration** — takes settings/deps as constructor args; the SHAPE of
   what it accepts isn't oceans-specific.
4. **Declared surface** — a deliberate public API (`__init__` exports).
5. **Honest docs** — docstrings describe THIS code.
6. **Portable tests** — tests could move with the package (no oceans fixtures).

Mechanical results (2026-07-24): check 1 is a clean sweep — zero `agents.`/`app.`
imports anywhere under `src/core/`. Env access is confined to `config.py`
(function-scoped `get_settings()`, not import-time). The gaps below are all the
subtler checks.

## Scorecard

| Component | Verdict | Gaps |
| --- | --- | --- |
| `core/context` | ✅ package-ready | verify summarizer-prompt sourcing (below) |
| `core/prompts` | 🟡 two small cuts | own exception; domain templates out |
| `core/llm` | 🟡 one design gap, one honesty failure | narrow credentials protocol; dead docstrings |
| `core/agents` | 🟡 umbrella by design | `dependencies.py` drags context+llm+prompts; stale docstrings; no tests |
| `core/config` | ❌ fails the litmus | oceans settings monolith wearing a core badge |
| `exceptions`, `logging_config` | (not deep-read) | small; verify at packaging time |

## Details

### `core/context` — ✅

Imports only itself. Defines its own `protocols.py` (Summarizer, TokenCounter,
ToolResultStore) — dependency inversion done properly. Curated `__init__` surface.
The ONLY component with unit tests, and they travel (`tests/core/context/`).

- [ ] Verify: `core/prompts/templates/summarizer/` exists — if `LLMSummarizer`
      reaches into another package's data at runtime that's a hidden coupling; if the
      prompt arrives injected, clean.

### `core/prompts` — 🟡

- [ ] Imports `core.exceptions.PromptError` — standalone package needs its own
      exception (trivial).
- [ ] **Domain templates ship inside the platform package**: `templates/oceans_agent/`
      lives in `core/prompts/`. Data/code conflation — the package ships the
      *registry*; consumers point it at *their* template dirs (a `search_path`
      constructor arg; may already half-exist — check).

### `core/llm` — 🟡

- [ ] `build_llm_registry(settings, …)` takes the whole oceans `Settings` but needs
      ~5 fields (anthropic/openai keys, hf token, ollama url, aws region/profile).
      Fat-interface coupling; standalone package wants a narrow credentials protocol.
      (Fails partly BECAUSE of the config monolith below.)
- [ ] **Check-5 failure:** docstrings reference `LLM_ROLE_CONFIG` in five places
      including the usage example — the symbol no longer exists anywhere in the code.
      Comment block `llm_registry.py:169-172` lists world-simulator-era consumers
      ("cluster agents", "logistics ReAct loop"). The docs describe a dead codebase.
- [ ] No tests.

### `core/agents` — 🟡

`node_executor` / `node_types` / `state_types` / `routing` are self-contained and
generic. But:

- [ ] `dependencies.py` imports context + llm + prompts — as a standalone package it
      drags all three. Packaging-boundary decision, not a bug: either ship one `core`
      package (fine), or move `AgentDependencies` out.
- [ ] World-simulator docstring headers (the X1 cosmetic item, confirmed still live).
- [ ] No tests.

### `core/config` — ❌ the finding

Not platform code — the **oceans settings monolith wearing a core badge**:
`postgres_url` defaults to the oceans DB, `image_root` → fluke JPEGs, `embedder_ver`
selects a whale-fluke embedder, `LANGSMITH_PROJECT = "oceans-simulator"` hardcoded,
module docstring says "world-simiulator.config". No *import* violates the layer rule,
but the *content* violates its spirit comprehensively.

Packageable shape: core owns at most a tiny base (the `get_settings` env-file
mechanics); **each app owns its `Settings`**. The maintainer-agent project therefore
writes its own `Settings` regardless — which it needed anyway.

## Net result for the maintainer-agent project

- `context` lifts today.
- `prompts` / `llm` / `agents` lift after small, well-understood cuts.
- `config` does not lift; new project writes its own Settings.
- Gaps are severable couplings, not structural rot — "reusable" survives due
  diligence with edits.

## The recursion payoff — every gap is a seed corpus rule

This analysis is the maintainer agent's job performed manually; harvest it:

- "Platform packages accept narrow protocols, not the app's Settings object" (llm)
- "Packages ship code; consumers ship data — templates, configs" (prompts)
- "Config CONTENT follows the layer rule, not just config imports" (config)
- "Docstrings referencing dead symbols are worse than no docstrings" (llm — also a
  good EVAL SEED: doc-rot is a thing the reviewer should flag)
- "A component isn't reusable until its tests can travel with it" (all but context)
