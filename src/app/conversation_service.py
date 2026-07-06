"""conversation_service.py — the application surface (walking skeleton).

The single object that owns the ocean agent's graph and exposes the conversation
lifecycle as methods. It is the transport-agnostic CORE: the HTTP server
(ocean_runner.py) and the no-server debug_driver both call the SAME methods —
FastAPI is pure transport that delegates here; debug_driver calls in-process.

Lifecycle scopes (nested — see CLAUDE.md):
  process → build_context() constructs ONE service at startup, shared by all requests
  session → create_session / initialize_context / end_session   (per conversation, keyed by session_id)
  turn    → ask()                                                (one question, one graph run)

STATELESS per session: session_id is a PARAMETER, never instance state. One service
instance is shared across every concurrent conversation, so per-session state must
live in the checkpointer (keyed by session_id), NEVER on this object. Holding "the
current conversation" here would silently commit to a server-owned loop.

Walking skeleton: ask() is fully wired. create_session / initialize_context /
end_session are honest STUBS — the shape (the seams) is laid down now; the bodies
land with their tickets (B1.5 durable sessions, B2 context assembly). Deliberate:
seams are expensive to change, method bodies are cheap.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from agent_chat.actions import Token, ToolCall
from agent_chat.protocols import TurnRequest

from agents.sandbox_agent.graph import build_sandbox_graph
from core.agents.dependencies import AgentDependencies

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def _text_chunks(content) -> list[str]:
    """Pull display text out of a streamed model chunk.

    Anthropic chunks are either a plain string (normal text) or a list of content
    blocks (when the model is also emitting tool calls). We only want the text.
    """
    if isinstance(content, str):
        return [content] if content else []
    out = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            if block.get("text"):
                out.append(block["text"])
    return out


class ConversationService:
    """The application surface — owns the graph, exposes the conversation lifecycle."""

    def __init__(self, caps: AgentDependencies, saver: BaseCheckpointSaver | None = None) -> None:
        # The graph is APPLICATION logic — it lives HERE, built from the capabilities
        # bag, not in a context struct beside framework infra (settings/registries/store).
        self._caps = caps
        self._graph = build_sandbox_graph(caps, saver=saver)

    # ── session lifecycle — STUBS (seams now, bodies later) ──────────────────

    async def create_session(self, session_id: str) -> None:
        """Session BIRTH: allocate per-conversation resources.

        Stub until B1.5 (durable checkpointer) gives session birth real work — e.g.
        registering the checkpointer thread. Maps to agent_chat's POST /sessions +
        the on_session_start hook.
        """
        return None

    async def initialize_context(self, session_id: str, seed: object | None = None) -> None:
        """Per-SESSION context bootstrap: seed prior-session context onto turn 1.

        Stub until B2 (context assembly). NOTE the scope: this is SESSION-scoped, NOT
        process-scoped — distinct from build_context(). Mirrors journal_agent's
        first_turn seed (recent_messages / user_profile).
        """
        return None

    async def end_session(self, session_id: str) -> None:
        """Session DEATH: teardown / archive.

        Stub until B1.5 gives sessions a durable end story (the 'death story' the app
        currently lacks). Maps to agent_chat's DELETE /sessions + on_session_end.
        """
        return None

    # ── turn lifecycle — fully wired (this is the old ocean_runner()) ─────────

    async def ask(self, request: TurnRequest) -> AsyncIterator:
        """Drive the ocean graph for ONE turn, yielding Frames.

        The user's message goes in as a HumanMessage; we stream the graph's events and
        translate the two a chat UI cares about: streamed model text (Token) and tool
        calls (ToolCall). session_id does double duty — seeded into OceanState (so
        node_executor stamps it on metrics/errors) AND passed as the checkpointer's
        thread_id, so successive turns with the same id share memory. Everything else
        (the ReAct loop) stays inside the graph.
        """
        # noqa on the input: OceanState is a pydantic model, so PyCharm rejects the plain
        # dict — but langgraph validates dict→state at runtime.
        async for event in self._graph.astream_events(  # noqa
            input={"messages": [HumanMessage(request.message)], "session_id": request.session_id},
            config={"configurable": {"thread_id": request.session_id}},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                # the model produced a piece of its answer — stream it to the bubble
                chunk = event["data"]["chunk"]
                for text in _text_chunks(chunk.content):
                    yield Token(text=text)

            elif kind == "on_tool_start":
                # a tool node fired — tell the UI which tool and with what input
                yield ToolCall(name=event["name"], args=event["data"].get("input", {}))

            elif kind == "on_tool_end":
                # the tool returned — report the result
                yield ToolCall(name=event["name"], result=event["data"].get("output"))

        # No AskHuman: the graph runs START→END each turn and never pauses mid-run.
        # Multi-turn memory comes from the checkpointer rehydrating prior state by
        # thread_id, NOT from HITL. Ask-human would be a separate, deliberate addition.
