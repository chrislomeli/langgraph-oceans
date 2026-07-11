"""state.py — the DURABLE shadow state the context module adds to the graph.

ONE GROUND TRUTH. `messages` (add_messages) is the sole, complete home of the raw
conversation — including full tool payloads. It is never mutated, trimmed, or
rewritten; it grows forever. Everything this module persists is DERIVED from it and
kept only because it is EXPENSIVE to recompute:

  - message_summaries  framed summary HumanMessage(s) — the output of an LLM call we
                       don't want to repeat. Append-only.
  - tool_calls         tool briefs (ToolBrief keyed by tool_call_id) — the aged-tool
                       index, merged as new regions are summarized.

Cheap-to-recompute things are NOT stored: the per-call view (manifest + summaries +
tail) is rebuilt every turn in view_builder and never written back. Because
`messages` is never mutated, the full tool payload always stays reachable there —
there is no second copy to keep in sync.

HOST STATE: mix these fields into the graph's state (oceans uses the pydantic
`OceanState(TracedState)`) — inherit `ContextStateFields` as a mixin or copy the
declarations. The Annotated reducers are read the same way on a pydantic state or a
TypedDict.
"""

from __future__ import annotations

from typing import Annotated, TypeVar

from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

T = TypeVar("T")

def update_dict(
    existing: dict[str, T],
    incoming: dict[str, T],
) -> dict[str, T]:
    """Reducer for `tool_calls`: merge the incoming briefs onto the existing dict.

    Keyed by tool_call_id (unique per call), so a merge only ever ADDS new briefs —
    it never overwrites a different call. Returns a new dict; never mutates `existing`.
    """
    if existing:
        merged = dict(existing)
        for key, value in incoming.items():
            merged[key] = value
    else:
        merged = incoming
    return merged



class ToolBrief(BaseModel):
    """A shrunk record of one tool call: what was asked + a truncated result.

    Created once per call by brief_tools when its region is summarized, then rendered
    into the leading manifest block by view_builder. `ordinal` is a gapless, monotonic
    global call-order number (see brief_tools) so the manifest can keep the NEWEST
    briefs under budget. name/query/summary are optional because a partial pair (only
    one half present in a slice) still produces a brief.
    """
    id: str
    ordinal: int
    name: str | None = None
    query:  dict | None = None
    summary: str | None = None



class ContextStateFields(BaseModel):
    """The exact field set the context module contributes to graph state.

    Mirror these into the host state (inherit as a mixin, or copy). These four persist
    across turns; everything else the module does is recomputed at call time.

      message_summaries           durable framed summary HumanMessages
                                  (add_messages reducer).
      tool_calls                  durable tool briefs, keyed by tool_call_id
                                  (update_dict reducer). See ToolBrief.
      last_processed_message_id   the bookmark; plain last-write (no reducer) —
                                  a single scalar advanced by one writer.
      token_count                 last actual total_tokens off the agent's reply;
                                  plain last-write, overwritten every hop.
    """

    # Pydantic V2 copies these mutable defaults per-instance (no default_factory needed).
    message_summaries: Annotated[list[HumanMessage], add_messages] = []
    tool_calls: Annotated[dict[str, ToolBrief], update_dict] = {}
    last_processed_message_id: str | None = None
    token_count: int = Field(default=0)

