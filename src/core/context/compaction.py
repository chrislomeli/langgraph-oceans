"""compaction.py — the short form of a tool result + the path back to the full one.

Reframed once tools and prose were separated. Compaction no longer swaps messages
inside the view tail. A tool result is now shown in exactly one of two shapes:

    HOT   (age < live_tail_size)   → full, structural, in the tail (live reasoning)
    aged  (age >= live_tail_size)  → a MANIFEST line: extract + ref (manifest.py)

So this module's job shrank to two things:
  1. produce the short extract a manifest line shows (`extract_of`), and
  2. resolve a ref back to the full payload (`resolve_full_tool_result`), which —
     because `messages` is never mutated — is always sitting right there in the log.

`ref_id` derives from `tool_call_id` so a manifest line points unambiguously back
to source. Compaction never summarizes (that's prose-only) and never mutates state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from core.context.protocols import ToolResultStore


def make_ref_id(tool_call_id: str) -> str:
    """`ref_{tool_call_id}` — derived, not random, so it stays traceable/stable."""
    raise NotImplementedError


def extract_of(message: ToolMessage, max_chars: int) -> str:
    """Short, human-usable extract of a tool result for a manifest line.

    First N chars / top-K structured fields (task dependent), truncated to
    `max_chars` (ContextPolicy.compaction_size_chars). Should usually answer the
    question on its own so the agent rarely needs to dereference. Errored results
    get an error-flavored extract. Pure — reads the message, returns a string.
    """
    raise NotImplementedError


def resolve_full_tool_result(
    messages: Sequence[BaseMessage],
    ref_id: str,
    store: ToolResultStore | None = None,
) -> Any:
    """Dereference a ref back to the full payload.

    Default path: parse the tool_call_id out of ref_id and return the content of the
    matching ToolMessage in `messages` — the full payload never left. If an external
    `store` is given (retention=EXTERNAL blobs), consult it first. Raises KeyError if
    the ref resolves to nothing (e.g. a retention=EPHEMERAL tombstone).
    """
    raise NotImplementedError


def build_get_full_tool_result_tool(store: ToolResultStore | None = None):
    """Factory for the agent tool `get_full_tool_result(ref_id)`.

    Returns a LangChain tool that calls resolve_full_tool_result against the thread's
    messages (via InjectedState) and the optional external `store`. Register it in
    the agent's toolset.

    Escape hatch, not the default read path: the manifest extract should usually
    suffice; dereferencing re-injects the full payload as a fresh (HOT) ToolMessage.
    """
    raise NotImplementedError
