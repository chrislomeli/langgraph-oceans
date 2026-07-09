"""manager.py — ContextManager: the one entry the summary node calls.

The real work of the module lands here. The summary node wired into the graph is a
thin wrapper (see below); this class holds the injected collaborators and runs the
per-turn dance.

PER-TURN DANCE:
  1. next_boundary carves the next PROSE slice from state — and it IS the gate:
     it returns None when there isn't a budget's worth of un-summarized prose yet.
  2. None => don't summarize; still build the view (summaries + manifest + tail),
     so existing chunks always apply. Unified path.
  3. Otherwise summarize the slice -> SummaryChunk(s), advance the bookmark.
  4. Always return the view on `llm_input_messages`; the only DURABLE writes are the
     new chunk(s) and the advanced bookmark.

Compaction/manifest are ephemeral and live inside build_view, so `prepare` never
returns tool state — only a chunk delta and the bookmark.

CHECKPOINT-RESUME SAFE: everything keys off last_processed_message_id read from
state, so a resumed thread continues from the bookmark rather than re-summarizing.

Wiring sketch (lives in the graph, not here):

    context = ContextManager(counter, policies)

    def summary_node(state: OceanState) -> dict:
        return context.prepare(state, policy_key="default", summarizer=summarizer)

    graph.add_node("summarize", summary_node)   # pre_model_hook / before agent
"""

from __future__ import annotations

from itertools import count
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from core.context.chunking import next_boundary
from core.context.protocols import Summarizer, TokenCounter
from core.context.retention import RetentionPolicy
from core.context.summarizer import summarize_messages
from core.context.view_builder import build_view


class ContextPolicy(BaseModel):
    """Tunables for the per-turn policy. Every open-question dial lives here.

    chunk_token_budget      target prose tokens per summarized slice (the gate).
    live_tail_size          trailing messages kept HOT: never summarized; tools
                            here stay full+structural.
    compaction_size_chars   max length of a manifest line's extract.
    view_token_ceiling      optional Q2 overflow guard for the assembled view;
                            summaries-of-summaries are a NON-GOAL, so this only
                            warns/raises rather than compressing further.
    """
    chunk_token_budget: int = Field(default=2000, ge=1)
    live_tail_size: int = Field(default=6, ge=0)
    compaction_size_chars: int = Field(default=1000, ge=0)
    view_token_ceiling: int | None = None

    # MEMORY axis (retention.py) — per-tool name→policy map, assembled by the app
    # from the tool ads. Empty ⇒ every tool is CHECKPOINT (full payload kept in
    # `messages` forever). This is a seam; the orchestrator doesn't apply it yet.
    tool_retention: dict[str, RetentionPolicy] = Field(default_factory=dict)


class ContextManager:
    """Assembles the model-call view from graph state, mutating nothing durable.

    Collaborators are injected (dependency inversion via protocols) so this class
    knows nothing of tiktoken, the LLM registry, or any storage backend. The app
    wires it once, alongside the existing AppContext deps.
    """

    def __init__(
            self,
            counter: TokenCounter,
            policies: Optional[dict[str, ContextPolicy]] = None,
    ) -> None:
        self.counter = counter
        self.policies = policies or dict(default=ContextPolicy())

    def _policy(self, policy_key: str) -> ContextPolicy:
        """Resolve a policy by key, or FAIL LOUD if it isn't configured.

        Deliberately no fallback: a missing policy is a wiring bug (the caller passed
        a key the composition root never wired). Defaulting to an arbitrary policy
        would silently summarize/build-view with the wrong budget — worse than a
        crash. Callers pass the correct key; if it's absent, say so clearly.
        """
        try:
            return self.policies[policy_key]
        except KeyError:
            raise KeyError(
                f"No context policy {policy_key!r}. Configured: {list(self.policies)}"
            ) from None

    def prepare(
            self, state: Any, policy_key: str, summarizer: Summarizer
    ) -> dict[str, Any]:
        """Run the per-turn dance; return a state-update dict for the graph.

        Reads from state: messages, summary_chunks, last_processed_message_id.

        Returns a dict that may contain:
          - "llm_input_messages": the ephemeral view (ALWAYS present).
          - "summary_chunks":     new chunk(s) — only when we summarized; the
                                  append_chunks reducer merges them in.
          - "last_processed_message_id": advanced bookmark — only when we summarized.

        Never returns anything under "messages" — the ground-truth log is untouched.
        Idempotent under checkpoint resume: no new chunk when there isn't a budget's
        worth of un-summarized prose.
        """
        policy = self._policy(policy_key)
        messages = state.messages
        bookmark = state.last_processed_message_id

        # 1. Carve the next prose slice. next_boundary IS the gate: it returns None
        #    when there isn't a budget's worth of un-summarized prose yet.
        boundary = next_boundary(
            messages,
            bookmark,
            self.counter,
            policy.chunk_token_budget,
            policy.live_tail_size,
        )

        # 2. Nothing to summarize this turn — no durable change. The agent node
        #    builds its own view (Option B), so prepare never builds a view.
        if boundary is None:
            return {}

        # 3. Summarize the slice into one or more chunks; advance the bookmark to the
        new_bookmark = boundary.end_message_id
        summarized_messages: list[HumanMessage] = summarize_messages(
            messages=messages,
            boundary=boundary,
            summarizer=summarizer,
            counter=self.counter)


        # 5. Return partials; LangGraph reducers merge them into OceanState.
        return {
            "summary_messages": summarized_messages,  # append_chunks reducer
            "last_processed_message_id": new_bookmark,  # overwrite
        }

    # proxy for build view with policy
    def build_view(self, state: Any, policy_key: str) -> list[BaseMessage]:
        policy = self._policy(policy_key)

        view = build_view(
            state.messages,
            state.summary_messages,
            state.last_processed_message_id,
            policy.live_tail_size,
            policy.compaction_size_chars
        )
        return view
