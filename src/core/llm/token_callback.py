"""
world-simulator.agents.commons.token_callback

LangChain callback that logs token usage and traces tool calls per LLM role.

One TokenUsageCallback instance is created per role in build_llm_registry
and attached to the chat model at construction time. Because it lives on
the model object, every call — plain invoke, with_structured_output, and
bind_tools — fires the hooks automatically with no changes at call sites.

Hooks implemented
─────────────────
  on_llm_end   : token counts from the API response (not estimated)
  on_tool_start: tool name + arguments the LLM chose to pass
  on_tool_end  : tool return value
  on_tool_error: exception raised inside the tool

Token count formats
───────────────────
  Anthropic : llm_output["usage"]["input_tokens"] / "output_tokens"
  OpenAI    : llm_output["token_usage"]["prompt_tokens"] / "completion_tokens"
"""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

_MAX_PENDING_TOOLS = 256


class TokenUsageCallback(BaseCallbackHandler):
    """Logs token usage and tool call lifecycle for one LLM role.

    Thread-safe for read (totals are only additive). Not designed for
    concurrent writes from multiple threads — sufficient for this
    single-threaded simulation loop.
    """

    def __init__(
        self,
        role: str,
        price_per_1m_input: float | None = None,
        price_per_1m_output: float | None = None,
    ) -> None:
        super().__init__()
        self.role = role
        self.total_input: int = 0
        self.total_output: int = 0
        self.call_count: int = 0
        self._price_per_1m_input = price_per_1m_input
        self._price_per_1m_output = price_per_1m_output
        self._tool_start_times: dict[UUID, float] = {}

    # ── LLM hook ──────────────────────────────────────────────────────────────

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        inp, out = self._extract_tokens(response)
        if inp == 0 and out == 0:
            return

        self.total_input += inp
        self.total_output += out
        self.call_count += 1

        logger.info(
            "[%s] call=%d  input=%d  output=%d  │  session total: %d in / %d out",
            self.role,
            self.call_count,
            inp,
            out,
            self.total_input,
            self.total_output,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.error(
            "[%s] llm_error  error=%s",
            self.role,
            error,
        )

    # ── Tool hooks ────────────────────────────────────────────────────────────
    #
    # run_id here is the tool's own run ID, distinct from the LLM run ID.
    # We use it as a key to match on_tool_start with on_tool_end so we can
    # compute elapsed time without any shared mutable state beyond the dict.

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        if len(self._tool_start_times) >= _MAX_PENDING_TOOLS:
            oldest = next(iter(self._tool_start_times))
            del self._tool_start_times[oldest]
            logger.warning(
                "[%s] _tool_start_times hit cap (%d); evicted oldest entry — "
                "on_tool_end may not have fired for run_id=%s",
                self.role,
                _MAX_PENDING_TOOLS,
                oldest,
            )
        self._tool_start_times[run_id] = perf_counter()
        logger.info(
            "TOOL:: [%s] tool_start  name=%s run_id=%s args=%s ",
            self.role,
            tool_name,
            run_id,
            _truncate(input_str),
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed_ms = self._pop_elapsed(run_id)
        logger.info(
            "[%s] tool_end  elapsed=%dms  output=%s",
            self.role,
            elapsed_ms,
            _truncate(output),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed_ms = self._pop_elapsed(run_id)
        logger.error(
            "[%s] tool_error  elapsed=%dms  error=%s",
            self.role,
            elapsed_ms,
            error,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pop_elapsed(self, run_id: UUID) -> int:
        start = self._tool_start_times.pop(run_id, None)
        return round((perf_counter() - start) * 1000) if start is not None else -1

    @staticmethod
    def _extract_tokens(response: LLMResult) -> tuple[int, int]:
        """Parse token counts from provider-specific llm_output shapes."""
        usage = response.llm_output or {}

        # Anthropic: {"usage": {"input_tokens": N, "output_tokens": M}}
        if "usage" in usage:
            u = usage["usage"]
            return u.get("input_tokens", 0), u.get("output_tokens", 0)

        # OpenAI: {"token_usage": {"prompt_tokens": N, "completion_tokens": M}}
        if "token_usage" in usage:
            u = usage["token_usage"]
            return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

        # Bedrock (ChatBedrockConverse) and modern LangChain attach usage to
        # the message as ``usage_metadata`` rather than in ``llm_output``.
        # Fall back to it so token accounting still works under Bedrock.
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if meta:
                        return meta.get("input_tokens", 0), meta.get("output_tokens", 0)
        except (AttributeError, TypeError):
            pass

        return 0, 0

    def report(self) -> dict:
        """Return a snapshot of usage totals for this role."""
        if (
            self._price_per_1m_input is not None
            and self._price_per_1m_output is not None
        ):
            cost = (
                self.total_input * self._price_per_1m_input
                + self.total_output * self._price_per_1m_output
            ) / 1_000_000
        else:
            cost = None
        return {
            "role": self.role,
            "calls": self.call_count,
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "estimated_cost_usd": cost,
        }

    def reset(self) -> None:
        """Reset counters — useful between simulation ticks in tests."""
        self.total_input = 0
        self.total_output = 0
        self.call_count = 0
        self._tool_start_times.clear()


# ── Module-level helper ───────────────────────────────────────────────────────

_TRUNCATE_AT = 200


def _truncate(value: Any, max_len: int = _TRUNCATE_AT) -> str:
    """Render value as a string, truncated so logs stay readable."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= max_len else text[:max_len] + "…"
