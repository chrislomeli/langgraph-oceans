"""chat_app.py — wire the sandbox_agent graph into the agent_chat streaming backend.

This is the ONE seam between your LangGraph graph and the chat front end. It does
exactly what `agents/sandbox_agent/graph.py::run_agent` does, but streaming:

    run_agent          →  ocean_runner (this file)
    ─────────             ─────────────────────────
    build_graph()      →  build_graph()                 (same graph, built once)
    graph.invoke(...)  →  graph.astream_events(...)      (watch it run, don't wait)
    return messages    →  yield Token / ToolCall ...     (emit Frames as it goes)

The graph still owns ALL the decisions (the ReAct loop, which tool to call, when to
stop). This runner only TRANSLATES the graph's event stream into Frames the UI shows.

Run it (from the repo root — needs src on the path and your env file for the API key):

    AI_ENV_FILE=.env uv run uvicorn app.chat_app:app --reload --app-dir src

Then point the existing React front end (examples/react-journal in the agent_chat
repo) at http://localhost:8000, send a question, and watch tokens stream.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage

from agent_chat import create_chat_app
from agent_chat.actions import Token, ToolCall
from agent_chat.protocols import TurnRequest

from agents.sandbox_agent.graph import build_graph
from config import get_settings

# Build the compiled graph once at startup (constructing the LLM client is cheap —
# no API call happens until a turn actually runs). Reused across every request.
get_settings().apply_langsmith()  # optional LangSmith tracing, same as the CLI
GRAPH = build_graph()


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


async def ocean_runner(request: TurnRequest) -> AsyncIterator:
    """Drive the ocean-conservation graph for one turn, yielding Frames.

    The user's message goes in as a HumanMessage; we stream the graph's events and
    translate the two that matter to a chat UI: streamed model text (Token) and
    tool calls (ToolCall). Everything else (routing, the ReAct loop) stays inside
    the graph and never surfaces here.
    """
    # Seed session_id into the graph state (OceanState inherits it from TracedState):
    # node_executor stamps it on every metric/error record, so a turn is traceable by
    # request. Same id will key the checkpointer's thread_id when multi-turn lands.
    async for event in GRAPH.astream_events(
        {"messages": [HumanMessage(request.message)], "session_id": request.session_id},
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
            # the tool returned — report the result (UI can show "echo → ...")
            yield ToolCall(name=event["name"], result=event["data"].get("output"))

    # Note: no AskHuman here. This graph answers in one turn and stops. Multi-turn
    # memory / ask-human comes later (add a checkpointer keyed on request.session_id).


app = create_chat_app(runner=ocean_runner, title="Ocean Conservation Agent")
