"""token_counter.py — pre-call token ESTIMATION for context budgeting.

Delegates to LangChain's `count_tokens_approximately` — the same fast, offline
heuristic LangGraph's own trimming/summarization use. It's more complete than the
old hand-rolled char/4 counter: it accounts for role, name, tool_calls,
tool_call_ids, and images, not just content characters. No tiktoken (it's OpenAI's
tokenizer and undercounts Claude), no network.

Two flavors of token accounting live in this package, and they are different jobs:
  - this file:          *pre-call* ESTIMATION — used to decide what to send (the gate).
  - token_callback.py:  *post-call* ACTUAL counts. For an exact number, prefer the
                        `usage_metadata` on a real model response over any estimate.

`HeuristicTokenCounter` structurally satisfies the `TokenCounter` Protocol in
`core/context/protocols.py`. It deliberately does NOT import that Protocol: the
dependency must run context -> llm, never back.
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
