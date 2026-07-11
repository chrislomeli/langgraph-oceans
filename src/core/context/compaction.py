"""compaction.py — turn aged tool results into short briefs.

Tools and prose are separate streams; this module owns the TOOL side. One job: shrink
each tool call in a boundary to a small ToolBrief. (There is no "resolve back to the
full result" path — `messages` is never mutated, so the full ToolMessage is always
still in the log if you ever need it; retrieval was deliberately forgone.)

WHEN briefs are made (the durable model): at SUMMARIZATION time. When prepare() finds
a boundary and summarizes its prose, it also calls `brief_tools()` on that SAME
boundary — the tools in the collapsed region have lost their position (their prose
neighbours just became a summary), so they move to a non-positional index. Each
tool's brief is created ONCE, there, and stored durably in state as

    tool_calls: dict[tool_call_id -> ToolBrief]

(merged by the update_dict reducer; rendered as the leading manifest block by
view_builder._render_manifest). HOT tools — still in the tail, still positional —
stay FULL; they are not briefed until their region is summarized.

Never summarizes (that's prose-only), never calls an LLM, never mutates state.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

from core.context.chunking import Boundary
from core.context.state import ToolBrief


def brief_tools(
    messages: Sequence[BaseMessage],
    boundary: Boundary,
    max_chars: int,
    base_ordinal: int = 0,
) -> dict[str, ToolBrief]:
    """Brief every tool call in `boundary` → {tool_call_id: brief}.

    The tool-side analog of summarizer.summarize_messages: same boundary, but the tool
    messages instead of the prose. Walk the AIMessage(tool_calls) + ToolMessage pairs
    in the slice and build, per tool_call_id, a uniform brief:
        {"name": <tool>, "query": <args>, "summary": <result, truncated to max_chars>}
    prepare() returns this as the `tool_calls` partial (dict-merge reducer).

    ORDINAL = global call order, GAPLESS and MONOTONIC across sweeps. Each NEW brief
    gets `base_ordinal + <count of new briefs so far this sweep>`; merging the other
    half of an existing pair does NOT consume an ordinal. The caller passes
    base_ordinal = len(state.tool_calls) (the append-only count), so a later sweep's
    briefs always outrank earlier ones — which is what _fit_manifest relies on to keep
    the NEWEST briefs under budget. (Using the slice index would reset to 0 each sweep
    and mis-order the manifest; see the ordinal bug.)

    Mechanical, no LLM. The boundary says WHICH tools to brief. Empty dict if the slice
    has no tool calls. Partial pairs (only one half in the slice) still get a brief —
    the missing side is left None — so every brief has the same shape.
    """
    slice_ = messages[boundary.start_index: boundary.end_index + 1]
    briefs: dict[str, ToolBrief] = {}
    next_ordinal = base_ordinal
    for message in slice_:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                existing = briefs.get(call["id"])
                if existing is None:
                    briefs[call["id"]] = ToolBrief(
                        id=call["id"], ordinal=next_ordinal, name=call["name"], query=call["args"]
                    )
                    next_ordinal += 1
                else:
                    briefs[call["id"]] = existing.model_copy(update={"name": call["name"], "query": call["args"]})

        elif isinstance(message, ToolMessage):
            summary = str(message.content)[:max_chars]
            existing = briefs.get(message.tool_call_id)
            if existing is None:
                briefs[message.tool_call_id] = ToolBrief(
                    id=message.tool_call_id, ordinal=next_ordinal, summary=summary
                )
                next_ordinal += 1
            else:
                briefs[message.tool_call_id] = existing.model_copy(update={"summary": summary})

    return briefs


