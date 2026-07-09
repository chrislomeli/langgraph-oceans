"""token_counter.py — Token estimation for context budgeting.

Salvaged from the single-shot ContextBuilder (core/context_builder.py) so the
context-management module can depend on token counting WITHOUT dragging in the
old prune-by-truncation logic, and without importing tiktoken itself.

Two flavors of token accounting live in this package, and they are different jobs:
  - this file (token_counter.py): *pre-call* ESTIMATION (fast heuristic, or
    tiktoken if a model name is given) — used to decide what to send.
  - token_callback.py:            *post-call* ACTUAL counts reported by the LLM.

`HeuristicTokenCounter` structurally satisfies the `TokenCounter` Protocol
declared in `core/context/protocols.py`. It deliberately does NOT import that
Protocol: the dependency must run context -> llm, never the reverse. The context
module owns the contract; this module owns one implementation of it.

This is the one file in the extraction that carries REAL bodies — it is working
utility being moved, not new design being stubbed.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage


def _content_len(message: BaseMessage) -> int:
    """Character length of a message's content, tolerant of list/dict content.

    AIMessages with tool_calls and multimodal messages can carry list content
    rather than a plain string; the old code assumed str and would raise. We
    coerce so the estimator never crashes on a tool-call turn.
    """
    content = message.content
    if isinstance(content, str):
        return len(content)
    return len(str(content))


class HeuristicTokenCounter:
    """Char-based token estimate, with an optional exact tiktoken path.

    Tunables (were class attrs on the old ContextBuilder):
        chars_per_token            — divisor for the fast heuristic (~4 for English).
        per_message_token_overhead — flat per-message role/formatting overhead.
        reply_priming_tokens       — added once per message list (assistant priming).
    """

    chars_per_token: int = 4
    per_message_token_overhead: int = 5
    reply_priming_tokens: int = 2

    def __init__(self, model: str | None = None) -> None:
        # If a model name is provided we try tiktoken; on any failure we fall
        # back to the heuristic. Kept optional so the default path has no
        # tiktoken dependency at call time.
        self.model = model

    # -- public surface: the TokenCounter contract -------------------------------

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        if self.model:
            try:
                return self._count_messages_tiktoken(messages, self.model)
            except Exception:
                pass
        return self._estimate_messages(messages)

    def count_text(self, text: str) -> int:
        if self.model:
            try:
                import tiktoken

                enc = tiktoken.encoding_for_model(self.model)
                return len(enc.encode(text))
            except Exception:
                pass
        return len(text) // self.chars_per_token

    # -- implementations ---------------------------------------------------------

    def _count_messages_tiktoken(self, messages: Sequence[BaseMessage], model: str) -> int:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        total = 0
        for message in messages:
            total += self.per_message_token_overhead
            total += len(enc.encode(str(message.content)))
        total += self.reply_priming_tokens
        return total

    def _estimate_messages(self, messages: Sequence[BaseMessage]) -> int:
        content_chars = sum(_content_len(m) for m in messages)
        overhead = len(messages) * self.per_message_token_overhead
        return (content_chars // self.chars_per_token) + overhead
