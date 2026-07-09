"""retention.py — per-tool MEMORY policy (durable, one-way). A SEAM, not yet wired.

A different axis from compaction. Keep them separate in your head:

  CONTEXT axis (compaction.py / view_builder.py): what the LLM sees per CALL.
    Ephemeral, recomputed every turn, fully reversible — the full payload always
    stays reachable in `messages`.
  MEMORY axis (this file): what the CHECKPOINT stores in Postgres, forever.
    Durable and ONE-WAY. Only relevant once a payload is COLD (already summarized,
    its conclusion captured) and big enough that the stored bytes matter.

`messages` never mutating is what buys context+traceability — but it's also what
makes the checkpoint grow without bound. Retention is the opt-out for the handful
of tools where that trade doesn't pay.

Three policies, declared PER TOOL (the tool author knows its output shape).
Default CHECKPOINT — behavior is unchanged until a tool opts out.

    CHECKPOINT  keep the full payload in `messages` forever (single ground truth).
                The default. Every other policy is an opt-out for large payloads.
    EXTERNAL    once COLD, move the full payload to a ToolResultStore and leave a
                stub+ref in `messages`. Checkpoint stays lean, RETRIEVAL STILL
                WORKS. For BIG but VALUABLE / non-reproducible results — live or
                temporal tools (vessel_traffic, web search) whose historical value
                a later re-call cannot reproduce.
    EPHEMERAL   once COLD, DROP the payload; leave a tombstone (ref + "not retained;
                re-call to refresh"). No retrieval. For BIG and DISPOSABLE results
                whose tool re-runs cheaply and deterministically (photo_id on a
                fixed image).

The dividing line between EXTERNAL and EPHEMERAL is one question: does re-calling
reproduce the result? If no, EXTERNAL — eviction would turn the traceability
invariant into a lie for exactly the data most likely to matter.

WHERE THIS WOULD PLUG IN (currently a no-op): the orchestrator applies retention at
COLD/summarization time — right after summarize_chunk captures the conclusion, the
full body's residual value drops to ~0. EXTERNAL/EPHEMERAL are the ONLY paths that
durably touch `messages`, via an add_messages upsert-by-id (same message id) that
replaces the content with a stub/tombstone. (Externalizing a known-huge tool at
tool-CALL time is the other valid trigger — same policy, earlier clock.)

DECLARE in the domain (tool ads) / WIRE in the app (assemble the name→policy map
into ContextConfig.tool_retention). core/ stays domain-agnostic: it knows only the
enum and the map it is handed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum

from langchain_core.messages import BaseMessage, ToolMessage

from core.context.protocols import ToolResultStore


class RetentionPolicy(str, Enum):
    CHECKPOINT = "checkpoint"
    EXTERNAL = "external"
    EPHEMERAL = "ephemeral"


def resolve_retention(
    tool_name: str | None,
    policies: Mapping[str, RetentionPolicy],
) -> RetentionPolicy:
    """Look up a tool's policy; unknown or None → CHECKPOINT (the safe default)."""
    raise NotImplementedError


def tool_name_of(message: ToolMessage, messages: Sequence[BaseMessage]) -> str | None:
    """Resolve which tool produced a ToolMessage.

    Prefer `message.name` when present; otherwise match `message.tool_call_id` back
    to the originating AIMessage.tool_calls entry in `messages` and read its name.
    """
    raise NotImplementedError


def externalize(message: ToolMessage, store: ToolResultStore) -> ToolMessage:
    """COLD + EXTERNAL: stash the full payload, return a stubbed replacement.

    store.put(ref, full); return a NEW ToolMessage with the SAME id and
    tool_call_id and content = extract + ref. Emitted back into `messages` via
    add_messages upsert (same id) — this DURABLY shrinks the checkpoint. Retrieval
    (resolve_full_tool_result) then finds the payload in the store.
    """
    raise NotImplementedError


def evict_to_tombstone(message: ToolMessage) -> ToolMessage:
    """COLD + EPHEMERAL: replace the payload with a tombstone. No store.

    Return a NEW ToolMessage, SAME id and tool_call_id, content =
    "<tool> result not retained (ref_...); re-call to refresh". Emitted back via
    add_messages upsert. After this the ref no longer resolves — the tombstone (and
    its manifest line, once we add the manifest) is the agent's only trace that the
    call happened.
    """
    raise NotImplementedError
