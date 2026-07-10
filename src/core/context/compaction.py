"""compaction.py — turn tool results into short briefs, and find the way back.

Tools and prose are separate streams; this module owns the TOOL side. Two jobs:
shrink a tool result to a short brief, and resolve a `tool_call_id` back to the full
result.

WHEN briefs are made (the durable model): at SUMMARIZATION time. When prepare() finds
a boundary and summarizes its prose, it also calls `brief_tools()` on that SAME
boundary — the tools in the collapsed region have lost their position (their prose
neighbours just became a summary), so they move to a non-positional index. Each
tool's brief is created ONCE, there, and stored durably as

    tool_summaries: dict[tool_call_id -> brief]

(merged on the state; rendered as the standing "tools you've called" block by
manifest.py). HOT tools — still in the tail, still positional — stay FULL; they are
not briefed until their region is summarized.

RETRIEVAL: `messages` is never mutated, so the full ToolMessage is always still in the
log. `resolve_full_tool_result` finds it by `tool_call_id` — no separate "ref"
indirection needed, since the brief is already keyed by `tool_call_id`. The optional
ToolResultStore seam is for LATER (when pruning removes the full message from the
log); until then it's unused.

Never summarizes (that's prose-only), never calls an LLM, never mutates state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

from core.context.chunking import Boundary
from core.context.protocols import ToolResultStore


def brief_tools(
    messages: Sequence[BaseMessage],
    boundary: Boundary,
    max_chars: int,
) -> dict[str, dict]:
    """Brief every tool call in `boundary` → {tool_call_id: brief}.

    The tool-side analog of summarizer.summarize_messages: same boundary, but the tool
    messages instead of the prose. Walk the AIMessage(tool_calls) + ToolMessage pairs
    in the slice and build, per tool_call_id, a uniform brief:
        {"name": <tool>, "query": <args>, "summary": <result, truncated to max_chars>}
    prepare() returns this as the `tool_summaries` partial (dict-merge reducer).

    Mechanical, no LLM. The boundary says WHICH tools to brief. Empty dict if the slice
    has no tool calls. Partial pairs (only one half in the slice) still get a brief —
    the missing side is left None — so every brief has the same three keys.
    """
    slice_ = messages[boundary.start_index: boundary.end_index + 1]
    briefs: dict[str, dict] = {}
    for message in slice_:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                brief = briefs.setdefault(call["id"], {"name": None, "query": None, "summary": None})
                brief["name"] = call["name"]
                brief["query"] = call["args"]
        elif isinstance(message, ToolMessage):
            brief = briefs.setdefault(message.tool_call_id, {"name": None, "query": None, "summary": None})
            brief["summary"] = str(message.content)[:max_chars]

    return briefs



def resolve_full_tool_result(
    messages: Sequence[BaseMessage],
    tool_call_id: str,
    store: ToolResultStore | None = None,
) -> Any:
    """Return the full tool result for a `tool_call_id`.

    Default path: scan `messages` for the ToolMessage whose tool_call_id matches and
    return its content — the log is never mutated, so it's always there. If an external
    `store` is provided (future: pruning moved the blob out of the log), consult it
    first. Raises KeyError if nothing resolves.
    """
    raise NotImplementedError


def build_get_full_tool_result_tool(store: ToolResultStore | None = None):
    """Factory for the agent tool `get_full_tool_result(tool_call_id)`.

    Returns a LangChain tool that calls resolve_full_tool_result against the thread's
    messages (via InjectedState) + the optional store. Register it in the agent toolset.

    Escape hatch, not the default read path: the brief in the manifest should usually
    answer the question. The tool_call_ids come straight from the manifest's keys.
    """
    raise NotImplementedError
