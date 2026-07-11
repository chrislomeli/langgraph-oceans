"""manager.py — ContextManager: the one entry the summary node calls.

The real work of the module lands here. The summary node wired into the graph is a
thin wrapper (see below); this class holds the injected collaborators and runs the
per-turn dance.

PER-TURN DANCE (prepare):
  1. next_boundary carves the next PROSE slice from state — and it IS the gate:
     it returns None when there isn't a budget's worth of un-summarized prose yet.
  2. None => nothing aged out this turn; return {} (no durable change). The agent
     node builds the view itself via build_view — prepare never builds a view.
  3. Otherwise: summarize the slice's prose into framed summary HumanMessage(s) AND
     brief its tool calls (brief_tools), then advance the bookmark.
  4. Return only the DURABLE deltas — message_summaries, tool_calls, and the new
     bookmark. `messages` is never touched.

View assembly is separate and ephemeral: the agent node calls build_view(state),
which recomputes the [manifest] + [summaries] + [tail] view every turn from this
state. prepare produces the durable inputs; build_view consumes them.

CHECKPOINT-RESUME SAFE: everything keys off last_processed_message_id read from
state, so a resumed thread continues from the bookmark rather than re-summarizing.

Wiring sketch (lives in the graph, not here):

    context = ContextManager(counter, policies)

    def summary_node(state: OceanState) -> dict:
        return context.prepare(state, policy_key="oceans_agent", summarizer=summarizer)

    graph.add_node("summarize", summary_node)   # before the agent node
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from core.context.chunking import next_boundary
from core.context.compaction import brief_tools
from core.context.protocols import Summarizer, TokenCounter
from core.context.retention import RetentionPolicy
from core.context.summarizer import summarize_messages
from core.context.view_builder import build_view


class ContextPolicy(BaseModel):
    """Tunables for the per-turn policy. Every open-question dial lives here.

    chunk_token_budget      target prose tokens per summarized slice (the gate).
                            SOFT: shapes the steady state, not a guarantee.
    live_tail_size          trailing messages kept HOT: never summarized; tools
                            here stay full+structural. SOFT.
    compaction_size_chars   max chars kept of each tool result in its brief
                            (brief_tools truncates the brief's summary to this).
    view_token_ceiling      the HARD ceiling on the assembled view. Enforced every
                            call by build_view's fitter cascade (shed manifest →
                            summaries → tail, current turn protected). The soft dials
                            above are the STARTING POINT; this is the backstop that
                            makes the view always fit. None ⇒ fitting OFF (no ceiling).
                            Ephemeral only — sheds from the outgoing list, never state.
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
        """Run the per-turn dance; return a durable state-update dict for the graph.

        Reads from state: messages, tool_calls, last_processed_message_id.

        Returns {} when nothing aged out this turn, otherwise the durable deltas:
          - "message_summaries":          new framed summary HumanMessage(s), merged
                                          by the add_messages reducer.
          - "tool_calls":                 new tool briefs, merged by update_dict.
          - "last_processed_message_id":  the advanced bookmark (plain overwrite).

        Never returns "messages" (the ground-truth log is untouched) and never returns
        the view (that's build_view, called separately by the agent node). Idempotent
        under checkpoint resume: no delta when there isn't a budget's worth of prose.
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

        # 3. Summarize the slice's prose into framed summary message(s); the new
        #    bookmark advances to the slice end so we never re-summarize it.
        new_bookmark = boundary.end_message_id
        summarized_messages: list[HumanMessage] = summarize_messages(
            messages=messages,
            boundary=boundary,
            summarizer=summarizer,
            counter=self.counter)

        # 4. Brief the slice's tool calls. Seed ordinals from the count already in
        #    state (briefs are append-only) so each sweep's briefs are strictly newer.
        compacted_tool_calls = brief_tools(
            messages=messages,
            boundary=boundary,
            max_chars=policy.compaction_size_chars,
            base_ordinal=len(state.tool_calls),
        )

        # 5. Return the durable deltas; LangGraph reducers merge them into the state.
        return {
            "tool_calls": compacted_tool_calls,          # update_dict reducer
            "message_summaries": summarized_messages,    # add_messages reducer
            "last_processed_message_id": new_bookmark,   # plain overwrite
        }

    # proxy for build view with policy
    def build_view(self, state: Any, policy_key: str | None = None) -> list[BaseMessage]:
        # Pass the injected counter + the policy's ceiling so build_view can enforce
        # the HARD limit. policy_key omitted ⇒ ceiling None ⇒ fitting stays OFF (the
        # current behavior); pass OCEANS_POLICY from the agent node to activate it.
        ceiling = self._policy(policy_key).view_token_ceiling if policy_key else None
        return build_view(
            messages=state.messages,
            message_summaries=state.message_summaries,
            tool_calls=state.tool_calls,
            last_processed_message_id=state.last_processed_message_id,
            counter=self.counter,
            view_token_ceiling=ceiling,
        )
