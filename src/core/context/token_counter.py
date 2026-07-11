"""token_counter.py — the module's default pre-call token estimator.

Delegates to LangChain's `count_tokens_approximately` — the same fast, offline
heuristic LangGraph's own trimming/summarization use. It accounts for role, name,
tool_calls, tool_call_ids, and images, not just content characters. No tiktoken (it's
OpenAI's tokenizer and undercounts Claude), no network.

This is the DEFAULT `ContextManager` uses when you don't inject your own — it's why
the module is self-contained (no dependency on the app's LLM layer). It structurally
satisfies the `TokenCounter` protocol; inject a different implementation there if you
have an exact tokenizer. For POST-call exact counts, prefer a response's
`usage_metadata` over any estimate — that's a separate concern, not this file's job.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately


class HeuristicTokenCounter:
    """count_messages / count_text via langchain_core's approximate counter."""

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        return count_tokens_approximately(messages)

    def count_text(self, text: str) -> int:
        # count_tokens_approximately is message-oriented; wrap the raw text so the
        # single estimator drives both methods (the ~3-token message overhead is
        # negligible for a summary-sized string).
        return count_tokens_approximately([HumanMessage(content=text)])
