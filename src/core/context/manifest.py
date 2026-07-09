"""manifest.py — the "tools you've called" index. A view-time PROJECTION.

Now the SOLE representation of aged tools (post stream-split). A tool is either HOT
(full + structural in the tail) or here — one compact line per call, derived fresh
each turn from the frozen `messages`, never stored:

    ref_call_abc · photo_id(image=H8) · "3 candidates, top #479"
    ref_call_def · vessel_traffic(bbox=…) · [not retained — re-call to refresh]

Three jobs one projection covers:
  1. lets the agent AVOID redundant calls ("I already looked that up");
  2. it's the WHOLE tool-traceability story — every call listed with its ref,
     independent of prose summaries (summaries no longer carry tool refs at all);
  3. it's the home for retention tombstones: an EPHEMERAL-evicted result is just a
     line flagged "not retained" — that flag IS the residual value the agent keeps.

Extract + ref come from compaction (extract_of / make_ref_id), so "short form of a
tool result" stays one function, used only here now.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, ToolMessage


def build_tool_manifest(
    messages: Sequence[BaseMessage],
    live_tail_size: int,
    max_chars: int,
    include_hot: bool = False,
) -> str:
    """Render the manifest block from `messages`. Pure projection; no state.

    Walks `messages` for tool calls (AIMessage.tool_calls paired to their ToolMessage
    by tool_call_id) and emits one line each, oldest first.

    Args:
        messages:       full log (input only).
        live_tail_size: HOT window size. HOT calls are already full+structural in the
                        tail, so by default they're omitted here to avoid duplication.
        max_chars:      max length of each line's extract (compaction.extract_of).
        include_hot:    include HOT calls too (a complete index) if True.

    Returns:
        A newline-joined block, or "" if there are no eligible tool calls. Folded
        into the view's top section after the COLD summaries by build_view.
    """
    raise NotImplementedError


def _manifest_line(tool_message: ToolMessage, tool_name: str | None, max_chars: int) -> str:
    """One line: `ref · tool_name(args?) · extract-or-tombstone`.

    ref = compaction.make_ref_id(tool_call_id); extract = compaction.extract_of(...,
    max_chars). Detects a retention tombstone (content set by
    retention.evict_to_tombstone) and renders "[not retained — re-call to refresh]"
    instead of an extract. Always includes the ref so the line is actionable.
    """
    raise NotImplementedError
