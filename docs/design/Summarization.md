Custom Summarization Node for LangGraph — Design Spec

Context

We're replacing/avoiding LangMem's prebuilt SummarizationNode with a hand-rolled
version, for transparency and to implement a few specific design choices that
the prebuilt node doesn't cleanly support. This is a LangGraph agent with a
standard messages list in state (add_messages reducer), including tool
calls (AIMessage.tool_calls) and their results (ToolMessage).

Core design principles (non-negotiable)


Never mutate state["messages"]. It is the permanent, complete,
ground-truth log. No RemoveMessage, no trimming, no rewriting. It grows
forever and is what gets checkpointed as history.
Summaries live in a separate "shadow" state field, not inline in
messages. The main model-calling node decides at call time whether to
send full messages, or a compressed view (summary + recent messages) —
that decision doesn't touch the source of truth.
Chunk boundaries are always contiguous ranges of messages, identified
by message id (every BaseMessage has one, auto-generated if not set).
No pairwise ID matching between human/AI turns — that's not a real
relationship in the data. Turn order is just list order.
A chunk boundary must never split a tool-call pair. If an AIMessage
with tool_calls falls inside a chunk, every matching ToolMessage
(matched via tool_call_id) must be in the same chunk. This is a hard
constraint, not a style preference — a split pair breaks the next LLM API
call outright (Anthropic/OpenAI both require every tool_calls entry to be
immediately followed by its result).
Tool results get compacted, not summarized. Compaction ≠ summarization:
the AIMessage/ToolMessage pair stays structurally intact (real
tool_calls, real tool_call_id) — only the ToolMessage.content shrinks
to a short extract + a reference pointer. The full tool payload is stored
externally and retrievable by that pointer. Only non-tool conversational
turns (human/AI prose) go through actual LLM summarization once they age
past the window.
Everything must be traceable back to source. Given a chunk summary, it
should be possible to resolve which original message.ids it covers and
pull the full, uncompressed messages back out of messages (they're still
there — see principle 1). Given a compacted tool result's reference ID, it
should be possible to retrieve the full original payload.


State schema additions

pythonclass SummaryChunk(TypedDict):
    id: str                      # this chunk's own id
    start_message_id: str        # first message id covered (inclusive)
    end_message_id: str          # last message id covered (inclusive)
    message_ids: list[str]       # all message ids covered, in order
    summary_text: str
    token_count_estimate: int
    created_at: str              # timestamp

class State(MessagesState):
    summary_chunks: list[SummaryChunk]   # ephemeral summary list; append-only
    last_processed_message_id: str | None  # bookmark: everything up to here
                                            # has already been considered for
                                            # chunking/summarization
    tool_result_refs: dict[str, Any]     # ref_id -> full original tool payload

summary_chunks is the "shadow summary list" — grows in parallel with
messages, never replaces it. Because it's a list of discrete chunks (not one
rolling blob), the agent's compressed view can be built by concatenating
chunk summaries + trailing raw messages, and any individual chunk can be
"expanded" back to its source messages via message_ids.

Components (suggested module breakdown)

1. chunking.py — boundary detection


Given messages and a starting point (last_processed_message_id), find
the next valid chunk boundary that:

respects a token budget (approximate token counter, pluggable — default
fast heuristic, optional exact tokenizer),
never ends on an AIMessage with tool_calls unless all matching
ToolMessages are included — extend the boundary forward until the pair
is complete,
never starts mid-pair either (shouldn't happen if the previous boundary
was chosen correctly, but validate defensively).



Provide a standalone validate_no_orphans(messages: list[BaseMessage]) -> list[str]
utility: collects all tool_call_ids from AIMessage.tool_calls and all
tool_call_ids from ToolMessages, returns the diff. Use this in tests and
as a runtime assertion after any boundary computation.


2. compaction.py — tool result compaction


Hook that runs on ToolMessage creation (or as a post-processing step right
after ToolNode runs, before the message re-enters the graph).
Stores the full tool result in state["tool_result_refs"] keyed by a
generated ref_id (tie it to tool_call_id for traceability — e.g.
ref_id = f"ref_{tool_call_id}").
Replaces ToolMessage.content with a short extract (first N chars / top-K
structured fields, task-dependent) + explicit reference, e.g.:
"3 results found. Top match: '<title>'. Full results: ref_call_abc123".
Provide a companion tool (get_full_tool_result(ref_id)) the agent can call
to dereference — this needs to be registered as a real tool in the agent's
toolset.
Decide (open question, see below): compact immediately on every tool
result, or only once a message ages past the chunk boundary. Recommend:
compact immediately for large/structured payloads (search results, file
contents), skip compaction for small payloads where compaction overhead
isn't worth it (define a size threshold).


3. summarizer.py — LLM summarization of non-tool chunks


Takes a validated chunk (contiguous, tool-boundary-safe) of messages.
Since tool results are already compacted (small), this mostly summarizes
human/AI prose turns — but must still handle chunks that contain
already-compacted tool pairs gracefully (pass them through / fold lightly
rather than re-summarizing something already short).
Use a structured prompt template, not freeform "summarize this" — dedicate
explicit fields (e.g. user intent, decisions made, open questions, key
facts/entities mentioned) so the summarizer can't silently drop something
by omission. This is a known failure mode of freeform summarization
(specific facts, IDs, names get lost) — structure mitigates it.
Output: a SummaryChunk as defined above, appended to
state["summary_chunks"].
Model choice should be pluggable/cheap — this doesn't need your main agent
model.


4. view_builder.py — compressed view for the model call


Given full messages + summary_chunks + last_processed_message_id,
build the actual list of messages to hand to the LLM this turn:
[SystemMessage(summary_of_all_chunks_concatenated), *messages[after last chunk boundary]].
This is the only place "summary" and "raw messages" actually combine — and
it's ephemeral, computed fresh each call, never written back to
state["messages"].


5. orchestrator.py — the node itself


Wired as pre_model_hook (or explicit node before your model-calling node).
Each turn:

Cheap check: token-count the delta since last_processed_message_id
(real, already-materialized content — not a prediction).
If under threshold: no-op, pass messages through as-is (or via
view_builder if chunks already exist from earlier).
If over threshold: find next valid chunk boundary (chunking.py),
summarize it (summarizer.py), append to summary_chunks, advance
last_processed_message_id.
Return the view (view_builder.py) for the model call.





Explicit edge cases to test


Chunk boundary lands exactly on an AIMessage with a single tool_calls
entry — boundary must extend to include the matching ToolMessage.
Chunk boundary lands on an AIMessage with multiple parallel
tool_calls — boundary must extend to include all matching
ToolMessages, not just the first.
A HumanMessage with no AIMessage reply yet (mid-turn) — must not be
swept into a chunk or dropped; this is normal in-progress state, not an
orphan.
Chunk boundary computed, but last_processed_message_id doesn't exist in
messages (shouldn't happen, but validate — could indicate a bug elsewhere
or a manually edited state).
Two chunks in summary_chunks with overlapping message_ids — should
never happen; add an assertion.
Tool result compaction when the tool call errored (no clean "result") —
decide how compaction handles error payloads (probably: store full error,
compact to a short error summary + ref, same as success case).
Re-running summarization on a thread that already has summary_chunks from
a previous session (checkpoint resume) — orchestrator must resume from
last_processed_message_id, not re-summarize from scratch.


Explicit non-goals


No attempt to summarize/compress tool payloads themselves via LLM — that's
compaction's job (structural shrink + pointer), not summarization's.
No pairwise ID linking between human/AI turns — chunk boundaries are
sufficient; don't build anything analogous to tool_call_id for
non-tool messages.
Not replacing LangGraph's add_messages reducer or checkpointing — this
layers on top of standard LangGraph state, not a replacement for it.


Open questions to resolve during implementation


Token counter: fast approximate (e.g. count_tokens_approximately-style
heuristic) vs. exact provider tokenizer — pick one as default, make it
swappable.
Compaction timing: immediate-always vs. size-threshold-gated (see
compaction.py above) — needs a concrete byte/token threshold.
Where does tool_result_refs actually live long-term — in-memory dict in
state (checkpointed with everything else, simplest) vs. external store
(DB/blob storage, needed if payloads are large or need to survive outside
the checkpoint)? Start with in-state dict, leave a seam to swap in an
external store later.
Summary chunk model: same model as the main agent, or a distinct
cheap/fast model bound separately? Recommend distinct.