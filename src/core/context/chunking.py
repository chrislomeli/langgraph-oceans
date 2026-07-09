"""chunking.py — Boundary detection for PROSE summarization. Pure, no I/O, no LLM.

Simplified once we separated the streams: tools are handled entirely by compaction
+ manifest and are NEVER summarized, so summarization sees prose only. That removes
the one hard part this module used to carry — tool-pair safety — from the boundary.

What's left is just a size decision:

  Walk the eligible region (bookmark → -live_tail_size), accumulate PROSE tokens
  until ~budget, and cut. No forward-walk-to-close-pairs, no orphan checks — there
  are no pairs in prose to split.

Pair-safety didn't vanish from the system, it RELOCATED: the only structural tool
pairs the view ever sends live in the HOT window, so `validate_no_orphans` now
guards that window's leading edge (used by view_builder), not the summary boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, ConfigDict

from core.context.protocols import TokenCounter


class Boundary(BaseModel):
    """A resolved prose-summarization boundary over `messages`.

    Frozen: a computed value object, not mutable state. The range is contiguous in
    `messages`; it MAY contain interspersed tool messages (they ride along under the
    advancing bookmark) but the summarizer will filter to prose. `message_ids` are
    the durable handles that go into the SummaryChunk.
    """

    model_config = ConfigDict(frozen=True)

    start_index: int
    end_index: int              # inclusive
    start_message_id: str
    end_message_id: str
    message_ids: list[str]


def index_after_bookmark(
    messages: Sequence[BaseMessage], bookmark_id: str | None
) -> int:
    """First eligible index — where the un-summarized region begins.

    None => a fresh thread, start at 0. Otherwise find the bookmark and start just
    after it. Scans backward because the bookmark sits near the tail. Raises if the
    id is given but absent — that's corrupted/edited state, and failing loud here is
    the whole reason we key off an id rather than a stored index.
    """
    if bookmark_id is None:
        return 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].id == bookmark_id:
            return i + 1
    raise ValueError(f"bookmark {bookmark_id!r} not found in messages")


def _is_prose(message: BaseMessage) -> bool:
    """True for conversational turns; False for tool requests/results.

    Tools drive the manifest, not the prose budget, so they count as ~0 here: a
    ToolMessage (a result) or an AIMessage that carries tool_calls (a request).
    Everything else — human turns, plain assistant text — is prose.
    """
    if isinstance(message, ToolMessage):
        return False
    if isinstance(message, AIMessage) and message.tool_calls:
        return False
    return True


def next_boundary(
    messages: Sequence[BaseMessage],
    after_message_id: str | None,
    counter: TokenCounter,
    budget: int,
    live_tail_size: int,
) -> Boundary | None:
    """Compute the next prose slice to summarize, or None if nothing is eligible.

    Args:
        messages:          full, ordered message log (never mutated).
        after_message_id:  bookmark; start AFTER this id. None => start at 0. Raise
                           if given but not present (corrupted/edited state).
        counter:           token estimator, applied to PROSE only.
        budget:            target prose tokens for the slice.
        live_tail_size:    trailing messages that are off-limits (the HOT fence).

    Returns:
        A contiguous Boundary starting just after the bookmark, growing until the
        PROSE it covers reaches ~budget or it reaches the HOT fence. Tool messages
        inside the range are counted as ~0 (they don't drive the prose budget) and
        are not summarized. None if there isn't a budget's worth of eligible prose
        yet — that's the "wait for more" case.

    Raises:
        ValueError: after_message_id given but not present in messages.

    No pair logic here anymore. A boundary that ends between an AIMessage(tool_calls)
    and its ToolMessage is FINE: neither half is rendered structurally from the
    bookmark — aged tools are manifest-only, and HOT tools are guarded separately.
    """
    start = index_after_bookmark(messages, after_message_id)
    fence = len(messages) - live_tail_size  # exclusive: [start, fence) is eligible
    if start >= fence:
        return None  # nothing outside the HOT window yet

    prose_tokens = 0
    end_index: int | None = None
    for i in range(start, fence):
        if _is_prose(messages[i]):
            prose_tokens += counter.count_messages([messages[i]])
        end_index = i
        if prose_tokens >= budget:
            break  # this prose message tipped us over — include it, stop here

    if prose_tokens < budget:
        return None  # not a budget's worth of prose yet — wait for more

    covered = messages[start : end_index + 1]
    return Boundary(
        start_index=start,
        end_index=end_index,
        start_message_id=covered[0].id,
        end_message_id=covered[-1].id,
        message_ids=[m.id for m in covered],
    )


def validate_no_orphans(messages: Sequence[BaseMessage]) -> list[str]:
    """Return tool_call_ids opened but not closed (or vice versa) within a slice.

    Collects every tool_call_id in AIMessage.tool_calls and every tool_call_id on a
    ToolMessage, and returns the symmetric difference. Empty == pair-complete.

    New home: view_builder uses this to verify the HOT window's leading edge is
    pair-complete (extend the window back to close any straddling pair). It is NO
    LONGER used on the summary boundary — summaries don't carry structural pairs.
    """
    raise NotImplementedError
