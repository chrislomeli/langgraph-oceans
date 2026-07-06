"""debug_driver.py — drive the ConversationService directly, no server, so you can breakpoint it.

The chat backend (uvicorn) is one caller of ConversationService: its lifespan builds the
service and the OceanRunner adapter forwards each turn to `service.ask`. This file is a
SECOND caller that skips the adapter entirely — it builds the service by hand and calls
`service.ask(request)` directly. Same service, same method, same breakpoints — minus HTTP,
SSE, and the protocol shim. That directness is the point: the core is callable without transport.

    uv run python -m app.debug_driver

To step through it: set a breakpoint inside `ConversationService.ask` (e.g. the `yield
Token(...)` line) — both this file and the server reach the exact same method — and run THIS
file under the PyCharm/VS Code debugger.
"""
import asyncio
from uuid import uuid4

from agent_chat.protocols import TurnRequest

from app.context import build_context
from app.conversation_service import ConversationService

# Edit this to ask different things. The [query photo at: <path>] marker is what makes the
# agent call photo_id; drop it and you'll get the no-tools answer instead. A second question
# that references the first exercises multi-turn memory (same session_id below).
QUESTIONS = [
    "Is this whale at risk from ship strikes?\n\n"
    "[query photo at: /Users/chrislomeli/Source/DATA/oceans/images/556/60028.jpg]",
    "Which individual number did you just identify, and what was its core range?",
]


async def conversation_loop(service: ConversationService, session_id: str):
    # Every question reuses the SAME session_id → same checkpointer thread → memory.
    # debug_driver plays the client here: it owns the turn loop and the session id.
    for q in QUESTIONS:
        request = TurnRequest(session_id=session_id, message=q, metadata={})
        async for frame in service.ask(request):
            print(f"{type(frame).__name__:10} {frame}")


async def main() -> None:
    # Same ConversationService the server builds — here we build it directly and call
    # service.ask ourselves (no OceanRunner adapter, no HTTP).
    service = build_context()
    await conversation_loop(service, session_id=str(uuid4()))


if __name__ == "__main__":
    asyncio.run(main())
