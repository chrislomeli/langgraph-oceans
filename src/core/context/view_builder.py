"""view_builder.py — Build the ephemeral message list for one model call.

Computed fresh every turn, never written back to state["messages"]. This is what
makes "never mutate; send a compressed view" hold.

THE LANGGRAPH SEAM. In a pre_model_hook the returned view goes out on the
`llm_input_messages` channel — used for THIS call only, not persisted. The
orchestrator returns `{"llm_input_messages": build_view(...)}`.

THE MODEL (prose/tools split + the INLINE-summary decision):

    The view is CHRONOLOGICAL, not top-heavy. Walk `messages` in order and emit:
      • COLD region (before the bookmark) → its summary chunk(s) rendered IN PLACE,
        as normal-salience framed messages (see _render_summaries), where the
        original turns sat. NOT hoisted into the system block.
      • tail after the bookmark:
          - PROSE is kept raw
          - HOT tool pairs (age < live_tail_size) are kept FULL + structural
          - AGED tool pairs are DROPPED — they're in the manifest, not here

    WHY inline, not a top system block: a summary is compressed HISTORY, not an
    instruction. Hoisting it into `system` over-weights old, low-relevance content
    (system = standing, always-attended, operator authority). Left in position it
    reads as history at age-appropriate salience. The tool MANIFEST is the opposite
    — an INDEX the agent looks things up in (standing reference), so it belongs in a
    leading/standing block. History → inline; index → standing.

    WIRE CAVEAT: an inline summary needs a valid role at send time — a custom
    ChatMessage(role="summary") is NOT wire-valid. Render each summary as a framed
    normal turn in its slot, e.g. HumanMessage("[Summary of earlier turns: …]"), for
    normal salience. (Opus 4.8 mid-conversation `system` messages keep position but
    reintroduce system-priority — prefer the framed turn unless you want elevation.)

PAIR-SAFETY IS NOW A SINGLE, LOCAL CHECK. The only structural tool pairs are in the
HOT window, so the one rule is: the HOT window must not START mid-pair. Extend its
leading edge back to include any straddling AIMessage(tool_calls); verify with
chunking.validate_no_orphans. The summary boundary no longer carries this concern.

ALWAYS TRACEABLE TO `messages`. Every aged tool is a manifest line carrying its ref;
resolve_full_tool_result walks the ref back to the untouched ToolMessage in the log.

UNIFIED PATH. build_view is ALWAYS the outgoing list; with no summaries and a short
tail it's just the raw messages.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from core.context.manifest import build_tool_manifest
from core.context.state import SummaryChunk


def build_view(
    messages: Sequence[BaseMessage],
    summary_chunks: Sequence[SummaryChunk],
    last_processed_message_id: str | None,
    live_tail_size: int,
    compaction_size_chars: int,
    # system_prompt: str | None = None,  -- let the caller handle their own system prompt
) -> list[BaseMessage]:
    """Assemble [top block] + [structural tail] for the LLM.

    Args:
        messages:                   full log (input only; never mutated).
        summary_chunks:             ordered COLD prose summaries.
        last_processed_message_id:  prose is summarized up to here; the tail is
                                    everything after it.
        live_tail_size:             trailing messages kept HOT (full, structural).
        compaction_size_chars:      max length of a manifest line's extract.

    Assembly (chronological — see THE MODEL above):
      1. leading STANDING block = the tool manifest
         (build_tool_manifest(messages, live_tail_size, compaction_size_chars)).
         The agent's own system prompt is prepended by the CALLER, not here (that's
         why there's no system_prompt param) — decide whether the caller folds the
         manifest into its system message or you emit it as a leading framed block.
      2. INLINE summaries = _render_summaries(chunks), the COLD region rendered in
         place as normal-salience messages.
      3. the structural tail = _structural_tail(...).

    Returns the message list. With no chunks and a short tail, this is the raw
    messages unchanged.
    """
    raise NotImplementedError


def _structural_tail(
    messages: Sequence[BaseMessage],
    tail_start_index: int,
    live_tail_size: int,
) -> list[BaseMessage]:
    """The tail after the bookmark, prepared for the structural stream.

    Build a new list from tail_start_index:
      - PROSE messages: pass through.
      - HOT tool messages (age = distance-from-end < live_tail_size): pass through
        FULL and structural.
      - AGED tool messages (age >= live_tail_size): DROP — they're represented by
        the manifest in the top block.
    Then ensure the HOT window doesn't start mid-pair: if the first kept tool
    message is a ToolMessage whose AIMessage(tool_calls) was dropped, extend back to
    include that AIMessage (or drop the ToolMessage too). Verify with
    chunking.validate_no_orphans. Purely functional — mutates nothing.

    Caveat to handle: dropping an aged AIMessage(tool_calls)+ToolMessage pair can
    leave two consecutive same-role prose messages; merge/coalesce them if the
    provider requires strict alternation.
    """
    raise NotImplementedError


def _render_summaries(summary_chunks: Sequence[SummaryChunk]) -> list[BaseMessage]:
    """Render each COLD chunk as a normal-salience framed message, oldest first — to
    sit IN PLACE where its source turns were, NOT concatenated into a system block.

    Returns list[BaseMessage] (note: NOT a str anymore — that was the old hoisted-
    to-system design). One framed message per chunk, e.g.
    HumanMessage(f"[Summary of earlier turns: {chunk.summary_text}]"), so the summary
    reads as history at its position. Consecutive same-role framed summaries are fine
    (providers merge them). Kept per-chunk so a chunk can later be 'expanded' back to
    source via its message_ids without touching this rendering.
    """
    raise NotImplementedError
