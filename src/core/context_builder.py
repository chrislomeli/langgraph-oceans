"""context_builder.py — Assemble the LLM message list from state and instructions.

ContextBuilder is the single place that decides what the LLM sees on each turn.
Given a ContextSpecification (produced by the intent classifier), it:

1. Truncates session messages, recent messages, and retrieved fragments
   to the counts specified in the instruction.
2. Estimates token usage (fast heuristic, or tiktoken if a model name is given).
3. Prunes in a fixed priority order if the context exceeds the budget:
      retrieved context → recent messages → session messages
4. Wraps the system prompt and optional retrieved context into a SystemMessage.
5. Returns [SystemMessage, ...recent, ...session] ready to pass to the LLM.

Raises ContextBuildError (or subclass) if a required prompt is missing or
the context is still too large after all pruning.
"""

import json
import logging
from typing import Any

import tiktoken
from langchain_core.messages import BaseMessage, SystemMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ContextBuildError(Exception):
    """Base exception for context-building failures."""


class MissingStateError(ContextBuildError):
    """A required state key or prompt is missing."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Required state key missing: {key}")


class ContextTooLargeError(ContextBuildError):
    """Context still exceeds the token budget after all pruning."""

    def __init__(self, tokens: int, budget: int):
        self.tokens = tokens
        self.budget = budget
        super().__init__(
            f"Context is {tokens} tokens, still exceeds budget of {budget} after full pruning"
        )


class ContextBuilder:
    """Assemble and budget-fit the message list sent to the LLM.

    Class-level tunables:
        threshold  — safety margin (0.2 = keep 20% headroom for the reply)
        max_tokens — hard ceiling on estimated context tokens
    """
    threshold: float = 0.2
    per_message_token_overhead: int = 5
    max_tokens: int = 8000
    chars_per_token: int = 4

    def __init__(self):
        pass

    def get_context(
        self,
        prompt: str,
        # instruction: ContextSpecification,
        session_messages: list[BaseMessage] | None = None,
        recent_messages: list[BaseMessage] | None = None,
        retrieved_fragments: list[Any] | None = None,
        # insights: list[Insight] | None = None,
        # claim_insights: list[SubjectSnapshot] | None = None,
    ) -> list[BaseMessage]:
        """Manage the tokens being used in a turn
        I think that data will come in from:
           - System message  -- should only contain static data - everything else should be a /skill or tool?
           - Last Human message
           - Messages history -- can be pruned?


        """
        # MESSAGES  todo - too blunt - we should only prune if required ?
        session_messages = session_messages[-instruction.last_k_session_messages:] \
            if session_messages and instruction.last_k_session_messages else []  # guard against zeros and Nones

        # # Deprecate
        # recent_messages = recent_messages[-instruction.last_k_recent_messages:]\
        #     if recent_messages and instruction.last_k_recent_messages else [] # guard against zeros and Nones

        # TOOL data (per tool)   todo - generalize this - assuming that any rag or sql data will come into this on a tool ?:?
        retrieved_fragments = [
            {"content": f.content, "tag": [t.tag for t in f.tags]}
            for f in retrieved_fragments[:instruction.top_k_retrieved_history]
        ] if retrieved_fragments and instruction.top_k_retrieved_history else []

        # # Deprecate
        # if claim_insights:
        #     retrieved_insights = [s.model_dump() for s in claim_insights]
        # else:
        #     retrieved_insights = sorted([{"label": i.label, "body": i.body, "verifier_score": i.verifier_score} for i in insights or []], key=lambda d: d['verifier_score'])


        # perform a calculation of the token count --
        # todo - this should not be hard coded -- but there are diferent types:
        #    - json
        #    - messages
        #    - strings

        #  todo - should we try to calculate this way - or get actual token counts and prune for the next call - or both
        #    - is this getting token counts? src/core/llm/token_callback.py
        effective_max = self.max_tokens * (1 - self.threshold)
        def calculate_tokens():
            count_prompt_tokens = self.count_string_tokens(prompt)
            count_retrieved_fragments_tokens = self.count_string_tokens(json.dumps(retrieved_fragments))
            count_retrieved_insights_tokens = self.count_string_tokens(json.dumps(retrieved_insights)) if retrieved_insights else 0
            count_recent_tokens = self.count_message_tokens(recent_messages)
            count_session_tokens = self.count_message_tokens(session_messages)
            _count_all_tokens = count_prompt_tokens + count_retrieved_fragments_tokens + count_retrieved_insights_tokens + count_recent_tokens + count_session_tokens
            _overage_tokens = _count_all_tokens - effective_max
            return _count_all_tokens, _overage_tokens


        #

        # if we are over - drop retrieved context and insights first (same pruning priority)
        count_all_tokens, overage_tokens = calculate_tokens()
        while overage_tokens > 0 and retrieved_fragments:
            removed = retrieved_fragments.pop()  # pop from the end — lowest score since sorted desc
            overage_tokens -= self.count_string_tokens(json.dumps(removed))

        while overage_tokens > 0 and retrieved_insights:
            removed = retrieved_insights.pop()  # pop from the end — lowest score since sorted desc
            overage_tokens -= self.count_string_tokens(json.dumps(removed))

        count_all_tokens, overage_tokens = calculate_tokens()
        while overage_tokens > 0 and recent_messages:
            removed = recent_messages.pop()  # pop from the end — lowest score since sorted desc
            overage_tokens -=  self.count_message_tokens([removed])

        count_all_tokens, overage_tokens = calculate_tokens()
        while overage_tokens > 0 and session_messages:
            removed = session_messages.pop()  # pop from the end — lowest score since sorted desc
            overage_tokens -=  self.count_message_tokens([removed])

        # if we've pruned everything we can and are still over, bail
        count_all_tokens, overage_tokens = calculate_tokens()
        if overage_tokens > 0:
            logger.debug(
                f"After removing session_messages we are still {overage_tokens} tokens too big - throw an exception")
            raise ContextTooLargeError(int(count_all_tokens), int(effective_max))

        # Construct the System message
        logger.debug(f"System context: \n{prompt}")

        if retrieved_fragments:
            rc = f"\n\n<retrieved_context>\n{json.dumps(retrieved_fragments, indent=2)}\n</retrieved_context>"
            logger.debug(f"Retrieved Context: \n{rc}")
            prompt += rc

        if retrieved_insights:
            prompt += f"\n\n<reflection_insights>\n{json.dumps(retrieved_insights, indent=2)}\n</reflection_insights>"

        system_message = SystemMessage(
            content=prompt
        )

        # log everything
        logger.debug(
            "Recent Messages: %s",
            [m.model_dump_json() for m in recent_messages]
        )
        logger.debug(
            "Session Messages: %s",
            [m.model_dump_json() for m in session_messages]
        )

        # Construct all messages
        messages = [system_message] + recent_messages + session_messages
        return messages

    def count_message_tokens(self, messages: list, model: str | None = None) -> int:
        """Count tokens: tiktoken if *model* is given, else fast char-based estimate."""
        if isinstance(model, str):
            try:
                return self._count_message_tokens_with_tiktoken(messages, model)
            except Exception:
                pass

        return self._estimate_message_tokens(messages)

    def count_string_tokens(self, content: str, model: str | None = None) -> int:
        """Count tokens for a raw string: tiktoken if *model* is given, else estimate."""
        if isinstance(model, str):
            try:
                enc = tiktoken.encoding_for_model(model)
                return len(enc.encode(content))
            except Exception:
                pass
        return self._estimate_tokens_from_string(content)

    def _count_message_tokens_with_tiktoken(self, messages: list, model: str | None = None) -> int:
        enc = tiktoken.encoding_for_model(model)
        total = 0
        for message in messages:
            total += self.per_message_token_overhead  # every message has overhead tokens
            total += len(enc.encode(message.content))
        total += 2  # reply priming tokens
        return total

    def _estimate_message_tokens(self, messages: list[BaseMessage]) -> int:
        content_chars = sum(len(m.content) for m in messages)  # count
        per_message_overhead = len(
            messages) * self.per_message_token_overhead  # overhead multiplier = 5 in tokens not the same as threshold - this just adds 5 tokens to every message
        return (content_chars // self.chars_per_token) + per_message_overhead

    def _estimate_tokens_from_string(self, content: str) -> int:
        content_chars = len(content)
        return content_chars // self.chars_per_token
