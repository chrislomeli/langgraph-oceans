"""state.py — the DURABLE shadow state the context module adds to the graph.

ONE GROUND TRUTH (revised after the tool-call walkthrough). `messages`
(add_messages) is the sole, complete home of everything — including full tool
payloads. It is never mutated, trimmed, or rewritten; it grows forever.

  - Compaction is EPHEMERAL: a cheap view-time shrink of large tool results,
    recomputed each turn in view_builder, never written back. So the full tool
    payload always stays in `messages` — there is no second copy to keep in sync.
  - Summarization is DURABLE: an LLM call we don't want to repeat, so its output
    is cached here as append-only `summary_chunks`.
  - `tool_result_refs` is therefore NOT a state field by default — full payloads
    live in `messages`. It only reappears if you opt into an external ToolResultStore
    (large blobs out of the checkpoint); see protocols.ToolResultStore.

That split — durable = expensive, ephemeral = cheap — is the whole reason only
these two fields persist.

HOST STATE: oceans' live state is the pydantic `OceanState(TracedState)`. Add the
two fields below to it — either inherit `ContextStateFields` as a mixin (both are
pydantic) or copy the two declarations. The `Annotated[..., append_chunks]`
reducer is read by LangGraph the same way on a pydantic state as on a TypedDict.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class SummaryChunk(BaseModel):
    """One contiguous, tool-boundary-safe slice of history, summarized to prose.

    Invariants (assert on construction / in tests):
      - message_ids is contiguous in `messages` order, start..end inclusive.
      - the slice never splits a tool-call pair (see chunking.validate_no_orphans).
      - no two chunks in summary_chunks share any message_id (no overlap).
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    summary_text: str                # rendered structured summary (summarizer.py)
                                     # not of the source messages it replaced
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ContextStateFields(BaseModel):
    """The exact field set the context module contributes to graph state.

    Mirror these into OceanState (inherit as a mixin, or copy). These persist across
    turns — everything else the module does is recomputed at call time.

      summary_messages            durable, framed summary HumanMessages
                                  (add_messages reducer — upsert/RemoveMessage).
      last_processed_message_id   the bookmark; plain last-write (no reducer) —
                                  a single scalar advanced by one writer.
      token_count                 last actual total_tokens off the agent's reply;
                                  plain last-write, overwritten every hop.
    """

    # Pydantic V2 handles `= []` safely without needing a default_factory
    summary_messages: Annotated[list[HumanMessage], add_messages] = []
    last_processed_message_id: str | None = None
    token_count: int = Field(default=0)

