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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from agent_chat import create_chat_app
from agent_chat.actions import RunnerFrame, Token, ToolCall
from agent_chat.protocols import TurnRequest

from app.context import AppContext, build_context


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


async def ocean_runner(request: TurnRequest, ctx: AppContext) -> AsyncIterator:
    """Drive the ocean-conservation graph for one turn, yielding Frames.

    The user's message goes in as a HumanMessage; we stream the graph's events and
    translate the two that matter to a chat UI: streamed model text (Token) and
    tool calls (ToolCall). Everything else (routing, the ReAct loop) stays inside
    the graph and never surfaces here.
    """
    # Seed session_id into the graph state (OceanState inherits it from TracedState):
    # node_executor stamps it on every me       tric/error record, so a turn is traceable by
    # request. Same id will key the checkpointer's thread_id when multi-turn lands.
    graph = ctx.graph
    # noqa on the input: OceanState is a pydantic model, so PyCharm rejects the plain
    # dict — but langgraph validates dict→state at runtime. Same limitation the graph
    # builder noqa's (graph.py: "PyCharm can't match TypedDict to StateT bound").
    async for event in graph.astream_events(  # noqa
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


class OceanRunner:
    """Binds the process-scoped AppContext to agent_chat's request-only TurnRunner.

    `create_chat_app` wants a `runner(request) -> AsyncIterator[Frame]`, but the
    translation logic (`ocean_runner`) also needs the ctx. This holds ctx as
    read-only instance state and delegates each turn to `ocean_runner`, so the
    framework never sees the context — it lives here, at the composition seam.

    ONE instance is shared across every concurrent request, so it must stay
    effectively immutable after startup: hold ctx (shared deps) here, NEVER
    per-turn state — that belongs in the graph/checkpointer, keyed by session_id.

    ctx is filled at startup, not import. The server's lifespan sets it after
    `build_context()`; `debug_runner` passes it straight to the constructor.
    """

    def __init__(self, ctx: AppContext | None = None) -> None:
        self.ctx = ctx

    def __call__(self, request: TurnRequest) -> AsyncIterator[RunnerFrame]:
        if self.ctx is None:
            raise RuntimeError(
                "OceanRunner has no AppContext — build_context() must run first "
                "(the server does this in its lifespan; debug_runner passes it in)."
            )
        return ocean_runner(request, self.ctx)


# A stable runner object exists at import so create_chat_app can wire it into its
# router now; its ctx is populated at startup by the lifespan below — so importing
# this module (e.g. under uvicorn --reload) does no heavy work.
runner = OceanRunner()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: build the process context ONCE and bind it to the shared runner.

    Teardown (after `yield`) — closing pools, etc. — lands here later.
    """
    runner.ctx = build_context()
    yield


app = create_chat_app(runner=runner, lifespan=lifespan, title="Ocean Conservation Agent")
