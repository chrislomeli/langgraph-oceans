# Maintainer Agent — high-level design & feasibility

> Founding doc for the next project, whiteboarded 2026-07-24 in the oceans repo.
> Moves to the new repo when it exists. When this disagrees with reality, fix this first.

## What it is

An agentic AI that performs the **standards-conformance slice** of a software project
Maintainer's job: it knows a project's architecture, constraints, and organizational
philosophy, and can review code (e.g. intern change-sets) for fit against those
standards — **conversationally**, not as a one-shot report.

Explicitly NOT: a general code-review tool (correctness/bugs/style linting). The focus
is team goals and standards. Also NOT the whole Maintainer job — triage, changelog,
release notes etc. are out of scope (the word "Maintainer" is a scope bomb; this is one
slice of it, done well).

Non-functional goals: ReAct agent, context + goal management, project "understanding",
portfolio-grade for a top-level AI architect — real enough to actually be used, with
every cut corner documented (see ledger below).

## Core decisions (settled)

1. **Standards are AI-bootstrapped, human-ratified.** The human is responsible but
   leans heavily on AI: ask an AI to infer standards from a repo, then heavy manual
   curation. Standards authoring is a *precursor project*, not part of the ReAct agent.
2. **Machine-first corpus.** The standards data model is designed for machine
   consumption (retrieval + judgment at review time). Human-readable export is a
   nice-to-have we will likely never build.
3. **Guinea pig: oceans.** General data model, one concrete corpus (bootstrapped from
   the oceans repo + its CLAUDE.md). Multi-repo generality is claimed only for the
   *model*, not proven — that would need a second repo (deferred).
4. **Manufactured intern diffs = the eval set.** Curated change-sets with known ground
   truth ("violates rules X, Y" / "clean"), including subtle violations (right code,
   wrong layer). An AI "intern simulator" generates them. This is how we *measure*
   whether the agent works.
5. **Conversational review, not one-shot.** The maintainer asks for a review, then
   interrogates it: "why did you flag that?", "show me the code that does it right",
   "I disagree." Findings are durable structured session state (finding → rule cited →
   evidence → status: open/discussed/waived), not just prose output.
6. **Standards refinement in-conversation, with a hard guardrail:** the agent never
   silently edits the corpus mid-chat. Questions like "is it OK to use a lambda
   when…?" resolve as: (a) corpus answers it → cite the rule; (b) genuine gap → agent
   drafts a **proposed amendment** with status `draft`; (c) human ratifies → status
   flips to `ratified`.
7. **Conversation proposes, command disposes.** Two mechanisms, different in kind:
   - **Commands as mutation verbs** — `/standards adopt STD-041` (deprecate, amend…).
     Corpus writes never depend on the LLM inferring intent; the command handler flips
     state deterministically. The LLM-judgment / deterministic-state-change boundary is
     a headline architecture decision of the project.
   - **Commands as context assembly** — `/review change-7` declares what the session
     is about: load the change-set, scope-select relevant standards, initialize
     findings state, set the goal. This is the B5-CTX "/skills-shaped context bundle"
     idea from the oceans roadmap, finally earning its keep. Other bundles fall out:
     `/standards audit`, `/compare`.
8. **Unit of work: a toolized change-set, not a real PR.** `get_change(id)` returns
   the code chunk + touched files from a fixture folder. It's a *seam*: real GitHub
   PRs could be swapped in behind the same interface later. No git/webhook plumbing
   in the MVP.
9. **Interface: chat UI (reuse the ocean_runner seam) + a CLI/debug-driver path** for
   breakpoints — the existing oceans two-entry-point pattern, verbatim.

## Two deliverables

1. **Precursor: the standards corpus + its data model.** An offline, AI-assisted,
   human-in-the-loop authoring effort. Deliverable = the curated corpus, NOT an
   autonomous inference pipeline (building that pipeline is the "more than one person"
   version of this project — refused).
2. **The reviewer agent.** The runtime ReAct system that consumes the corpus,
   examines change-sets, judges fit, defends its findings across turns, and drafts
   amendments. This is the part that must "really work."

## Standards data model — requirements gathered so far

(Design note to be written before implementation — first component.)

- Rule = statement + **applicability scope** (when does this rule come into play —
  required for per-diff selection; without it we're stuffing the whole handbook into
  context) + **evidence anchors** (real repo code embodying it; ideally a violation
  example too — "violates ARCH-3, compare `src/app/context.py`" is the
  portfolio-grade behavior).
- **Tiers**: checkable rules vs. prose philosophy ("explicitness over magic") — the
  agent uses them differently.
- **Stable rule IDs**, **status** (draft / ratified / deprecated), **provenance**
  ("added 2026-07-24, from review session #12"). Required by the amendment flow and
  wanted anyway.

## Tool surface (sketch)

A diff alone is often not judgeable ("is this in the right layer?" needs surrounding
code). The interesting ReAct behavior is the agent *deciding to go look*:

- `get_change(id)` — the change-set fixture
- `read_file` / `search_code` — repo access for context-gathering
- `get_standards(scope)` — scoped corpus retrieval
- `propose_standard` — write path, always lands as `draft`
- (`ratify` is NOT a tool — it's a deterministic command handler)

## Reuse from oceans (~half the infrastructure exists, verified)

- Injection spine: `build_context()` → `AppContext` composition root
- `core/`: LLMRegistry, PromptRegistry, agent-graph framework (node_executor,
  node_types, metrics, AgentDependencies)
- B2.0 context module (summarize + brief + fitted view, prepare-at-call-time) —
  the long interrogation sessions are *why it exists*
- Durable `AsyncPostgresSaver` checkpointer — multi-turn review state
- Runner seam (astream_events → Token/ToolCall frames) + chat UI + debug driver

Due diligence: `docs/design/reuse-gap-analysis.md` (2026-07-24) — packageable litmus
run against all of `core/`. Result: `context` lifts today; `prompts`/`llm`/`agents`
lift after small cuts; `config` fails (oceans monolith — new project writes its own
Settings). Gaps are severable couplings, not structural rot.

Genuinely new work: standards data model, 3–4 tools, the corpus, the eval set,
findings-as-state in the graph state model, the `/command` dispatch layer in the
runner seam, prompt/judgment tuning.

## Cut-corners ledger (deliberate, documented)

1. No GitHub/PR integration — change-sets are toolized fixtures (a seam, one sentence
   to defend).
2. Standards bootstrap is interactive AI-assisted authoring, not an autonomous
   pipeline.
3. Generality: general data model, one concrete corpus; second-repo validation
   deferred.
4. Security / auth / multi-user: none, stated up front (not a product).

None of these undermine "actually usable" — the cuts are in delivery mechanics, not
in the reasoning.

## Where the real risk lives (not buildable-away)

1. **Judgment quality.** Failure modes: sycophancy ("looks good!") and nitpicking
   (style noise flagged as architecture violations). "Works but lives in the wrong
   layer" is genuinely hard. Defense: the ground-truth eval set. Budget: the longest,
   least predictable phase.
2. **Applicability retrieval.** Selecting *which* standards a change-set puts in
   play. Sloppy → missed violations or rule-drowning. Primarily a data-model problem
   (the scope field), which is why the precursor project matters.
3. **Conversational sycophancy under pushback.** The agent must hold a correct
   finding when the user pushes back, and concede when wrong. Adds a second eval
   dimension: not just "does it catch violations" but "does it hold/concede
   appropriately."

## Data-model whiteboard — settled 2026-07-24 (details go in the data-model design note)

- **Four kinds, one model.** Corpus entries are NOT homogeneous rules; kinds:
  `rule` (prescribed/proscribed + evidence anchors), `registry` (inventory + standing
  policy, e.g. "extend, don't reinvent"), `architecture` (layer map + dependency
  constraints), `memo` (cut-corner decisions + revisit condition; can graduate into a
  rule). Single entity with a `kind` discriminator and per-kind body — split only when
  a kind's body genuinely won't fit ("single model until it breaks").
- Common envelope: `id, kind, tier (rule|philosophy), scope, status, provenance,
  rationale`.
- **Scope = two selectors, OR'd:** `paths` (globs; mechanical) + `applies_when`
  (one-line NL predicate). Tags deliberately skipped for now (open: revisit).
- **"Scope-vs-ask" is not corpus** — judging a diff against what was requested needs
  an `intent` field on the change-set fixture (add from day one; can't retrofit into
  the eval set later).
- **Presence model (answers "right data at the right time"):**
  - Tier 0, always in context: corpus INDEX (id + one-liner for every ratified entry,
    ~3–4K tokens at ~60 entries) + architecture map. Agent has *awareness* of
    everything, *possession* of little.
  - Tier 1, hydrated at `/review` time: cheap triage LLM pass reads diff summary +
    index → returns applicable IDs → full bodies loaded. Two-stage: cheap model
    selects, big model judges.
  - Tier 2, on-demand: `get_standard(id)` etc. — a triage miss degrades to one extra
    tool call, not a wrong review (agent can always see the index).
  - Law: **assemble what's enumerable (corpus), fetch what's discoverable (repo).**
  - Same shape as the B2.0 tool-brief pattern and Claude Code's skill listing.
- **Storage: files (YAML/JSON dir) or single SQLite table — NO graph/vector DB at
  this scale.** PROVISIONAL: revisit if the drafted schema turns out heavily
  cross-referential. Scale risk is confined to Tier 0 (index size); its replacement
  (embedding retrieval over one-liners) is known and boring.
- **Dry-run trace validated the flow** (change #7, "add retry handling"): /review
  assembles ~10K ctx (system + arch map + index + 4 hydrated rules + diff/intent) →
  agent makes 3–4 targeted repo calls (read_file, search_code for prior art,
  self-serve get_standard) → findings cite rule ID + anchor + offending lines +
  prescribed alternative. Eval item: does the agent check the registry for prior art
  unprompted?

## Checklist discipline — settled 2026-07-24 ("do-confirm, not read-do")

The maintainer's checklist practice (bird's-eye list, lowers cognitive load, raises
completeness, "register a new thing" when one appears) maps onto existing parts: the
checklist IS the corpus index; registering = `/standards adopt`. The open risk was:
does giving the agent a checklist stop it from thinking? Answer: it depends entirely
on WHERE the checklist sits. Aviation framing: **read-do** (script drives the actions)
produces checklist theater in LLMs — mechanical item-walking, shallow per-item
glances, blind between items. **Do-confirm** (judge freely, then confirm coverage)
amplifies judgment instead of replacing it. Commitments:

1. **The checklist never drives; it gates.** The ReAct loop reviews freely — triaged
   rules are context, not an agenda. No "go through the rules one by one" anywhere in
   the prompt.
2. **Coverage gate at the end.** Before emitting findings, sweep the triaged rules:
   explicit per-rule verdict `violated | clean | not-applicable-because` → recorded in
   the findings schema as a **coverage record**. Completeness + anti-sycophancy:
   "clean" is an affirmative, interrogable claim, not an absence.
3. **Floor, not ceiling.** Findings outside the corpus are legitimate → tagged
   `unregistered-concern` ("this bothers me; no rule covers it"). Feeds the amendment
   pipeline directly: concern → maintainer agrees → `/standards adopt` → checklist
   grows. The agent becomes a contributor to the checklist, not just its executor.

**Open design option — split into two agents/nodes:** the coverage gate may be a
SEPARATE pass from the reviewer (fresh context, reads the diff + the draft findings +
the index, hunts for what the reviewer missed and rubber-stamped "clean"s). A
same-context gate inherits the reviewer's blind spots and self-consistency bias; a
second judge doesn't. Costs one more LLM call per review. Decide in the graph-design
note (it's a node-topology question — reviewer node → gate node → respond).

**Known failure mode + its eval:** the gate itself can rot into rubber-stamp theater.
Eval additions: seed violations (a) covered by a rule triage didn't hydrate (tests
floor + self-serve), (b) covered by NO rule at all (tests whether
`unregistered-concern` ever fires — if it never fires across the eval set, we built a
checklist executor and we'll know it).

Principle for the design note: **structure placed after judgment amplifies it;
structure placed before judgment replaces it.**

## Options kept open (not MVP; designed-for, not built)

- **Cross-repo evidence anchors.** "Clean" = our OWN vetted projects. Anchors get a
  `repo` field NOW (free); later, anchors may point into sibling reference repos —
  solves the young-repo "no prior art to cite" problem. Curation stays at the anchor
  level (repo-level "clean" is too coarse: every repo has non-exemplary corners).
  Approved-repos list is itself a `registry` entry.
- **Semantic code search** over own/approved repos, behind the existing `search_code`
  seam (grep now, hybrid later) — fixes vocabulary mismatch (intern's
  `reattempt_handler` vs. sanctioned `retry`). Hard boundary: **similarity FETCHES
  candidates, the LLM JUDGES conformance** — violations are near-duplicates of
  exemplars, cosine cannot discriminate them. No bulk RAG over raw repos, ever
  (un-vetted claims through the side door).
- **Whole-project compliance audit** ("how compliant is repo X; outline suspect
  parts") — same judgment core pointed at enumerated components instead of a diff.
  Design constraint TODAY: keep the judgment core separate from the
  "what-am-I-judging" input side, and audit mode falls out later.

## Data-shape assumptions & storage posture — settled 2026-07-24

Available engines: graph DB, SQL, Elasticsearch, Redis, vector/RAG. Posture: **no
engine is an architecture commitment; files-plus-index is merely the first repository
implementation.**

### Inventory: object → shape → today → future pressure → answering store

| Object | Shape | Access today | Future pressure | Store |
| --- | --- | --- | --- | --- |
| Corpus entries | docs w/ envelope + typed body + light cross-refs | load-all index · get-by-id · filter-by-status | refs become traversals; cardinality ≫ context | SQL, or graph if refs dominate |
| Corpus index | derived projection (id + one-liner) | always in context | > ~300 entries | RAG over one-liners |
| Evidence anchors | pointers: repo + path + lines (+ commit pin) | embedded in entries | anchor rot; cross-repo | stays embedded; validation job |
| Change-sets / eval fixtures | diff + intent + files + ground truth | get-by-id | more of them | files forever |
| Findings / sessions | structured state, statuses, links to rules/change-sets | inside LangGraph checkpointer | cross-session queries ("how often is ARCH-1 violated?") | SQL, outside checkpoints |
| Provenance / lifecycle | append-only events (drafted→ratified→deprecated, by whom, session #) | read per-entry | audit trails, corpus history | event log (SQL or in-entry) |
| Repo code | not stored — live via tools | grep via search_code | vocabulary mismatch; multi-repo | vector behind existing seam |
| Redis | — | — | caching triage results, session scratch | perf tool, never data-model |
| Elasticsearch | — | — | no identified job at any horizon | none |

### Explicit assumptions (what "YAML dir / SQLite" was silently betting on)

1. Cardinality ~10² corpus entries — THE load-bearing assumption behind
   index-always-in-context.
2. Cross-refs are by-ID and shallow — the LLM resolves them by reading; no store-side
   traversal. (Breaks if the drafted schema turns out densely relational — provisional
   flag already standing.)
3. Reads are load-all + get-by-id; the LLM is the query engine over the index.
4. One human-gated writer, low write rate — no concurrency story needed.
5. Findings analytics "don't exist yet" — ALREADY CREAKING; see below.

### The two protections (cheap now, brutal to retrofit)

1. **The agent never sees a store — only tools, and tools call a repository
   interface.** `get_standard(id)`, `list_index()`, `record_finding(...)` are
   engine-agnostic contracts; backends swap beneath them like `get_change` /
   `search_code`. Composition root picks the implementation (injection-spine
   discipline). NOTE: oceans did NOT wrap its data sources — this is a lesson carried
   forward, done from day one here.
2. **Record history, not just state, from day one.** Every store migration in the
   table is mechanical EXCEPT provenance never written down. Lifecycle changes are
   append-only events from the first ratify (even if the log is a list inside each
   YAML entry). Stable IDs are part of the same promise.

### Promoted to design now

**Findings get their own store-facing write path** (via the repository interface;
likely the same Postgres as the checkpointer) rather than living only in graph state.
Cross-session queries are implied by provenance ("from review session #12") and by
eval work — not a someday-maybe.

## Scope tripwires (renegotiate feasibility before adding ANY of these)

Autonomous standards inference · real GitHub integration · multi-repo generality ·
maintainer tasks beyond review (triage, changelog, release notes).

## Verdict

Feasible for one person — not comfortably, but the discomfort is in the right place
(judgment tuning, not infrastructure). Roughly half the plumbing already exists in
oceans and is verified.

## Next step

Design note for the **standards data model** — first component, everything downstream
consumes it. Per project process: mid-level design note before building.