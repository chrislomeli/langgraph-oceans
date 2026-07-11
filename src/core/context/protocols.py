"""protocols.py — The interfaces the context module needs from the outside.

This file is the anti-pollution wall. The context-management module declares
here WHAT it needs (count tokens, summarize a chunk, store/fetch a tool payload)
without knowing HOW any of it is done. Concrete implementations live elsewhere
(core/llm for the counter, the app wiring for the summarizer, this package or an
external store for the ref store) and are injected into ContextManager.

Because these are `Protocol`s, implementers satisfy them structurally — they do
NOT import from this file. Dependency therefore runs consumer <- implementer,
never the reverse. That is the whole point: the module stays free of tiktoken,
of your LLM registry, and of any storage backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage


@runtime_checkable
class TokenCounter(Protocol):
    """Pre-call token estimation. Satisfied by core.llm.HeuristicTokenCounter."""

    def count_messages(self, messages: Sequence[BaseMessage]) -> int: ...

    def count_text(self, text: str) -> int: ...


@runtime_checkable
class Summarizer(Protocol):
    """Turns a validated chunk of messages into summary prose.

    Injected so the module never binds to a specific model. Per the design's
    open question #4, this SHOULD be a distinct cheap/fast model, not the main
    agent model. The implementation lives in the app wiring, not here.
    """

    def summarize(self, messages: list[BaseMessage]) -> list[str]: ...


@runtime_checkable
class ToolResultStore(Protocol):
    """OPTIONAL external home for full tool payloads. Not wired yet.

    Not used by default: because `messages` is never mutated, the full tool payload
    always stays in the log — no store is required to reach it.

    This Protocol is the seam for the MEMORY axis (retention.py): the case where you
    want large blobs OUT of the checkpoint (DB/blob storage). retention.externalize
    would put() the payload here and leave a stub in `messages`. Until retention is
    wired, nothing consults it.
    """

    def put(self, ref_id: str, payload: Any) -> None: ...

    def get(self, ref_id: str) -> Any | None: ...
