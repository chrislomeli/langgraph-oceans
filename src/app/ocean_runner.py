"""ocean_runner.py — the HTTP transport seam + the TurnRunner protocol adapter.

FastAPI here is PURE TRANSPORT. All conversation logic lives in ConversationService
(app/conversation_service.py); this module only:
  1. adapts ConversationService.ask to agent_chat's request-only TurnRunner protocol
     (OceanRunner), and
  2. builds the FastAPI app + startup lifespan.

The same ConversationService is called directly (no HTTP) by app/debug_driver.py —
that's the whole point of keeping this layer logic-free.

    AI_ENV_FILE=.env uv run uvicorn app.ocean_runner:app --reload --app-dir src
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent_chat import create_chat_app
from agent_chat.actions import RunnerFrame
from agent_chat.protocols import TurnRequest

from app.context import build_context
from app.conversation_service import ConversationService


class OceanRunner:
    """Adapts ConversationService to agent_chat's request-only TurnRunner.

    `create_chat_app` wants a `runner(request) -> AsyncIterator[Frame]`. The service
    exposes `ask(request)` (plus session-lifecycle methods a single-turn protocol never
    sees). This object holds the process-scoped service and forwards each turn to
    `service.ask`, so the framework only ever sees one callable — the lifecycle surface
    stays behind this seam.

    ONE instance is shared across every concurrent request; it holds only the shared
    service (never per-turn state — that lives in the checkpointer, keyed by session_id).

    The service is filled at STARTUP, not import: the server's lifespan sets it after
    build_context(). Importing this module (e.g. under uvicorn --reload) does no heavy work.
    """

    def __init__(self, service: ConversationService | None = None) -> None:
        self.service = service

    def __call__(self, request: TurnRequest) -> AsyncIterator[RunnerFrame]:
        if self.service is None:
            raise RuntimeError(
                "OceanRunner has no ConversationService — build_context() must run first "
                "(the server does this in its lifespan)."
            )
        return self.service.ask(request)


# Stable runner object at import so create_chat_app can wire it now; its service is
# populated at startup by the lifespan below — so importing this module does no heavy work.
runner = OceanRunner()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Process-lifecycle STARTUP: build the service ONCE, bind it to the shared runner.

    Teardown (after `yield`) — closing pools when B1.5 lands a Postgres saver — goes here.
    """
    runner.service = build_context()
    yield


app = create_chat_app(runner=runner, lifespan=lifespan, title="Ocean Conservation Agent")
