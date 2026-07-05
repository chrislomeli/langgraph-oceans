"""debug_runner.py — drive the OceanRunner directly, no server, so you can breakpoint it.

The chat backend (uvicorn) is one caller of OceanRunner: it binds ctx via a lifespan.
This file is a second caller: it hands ctx straight to OceanRunner(ctx), builds a
TurnRequest by hand, and pulls frames out with an `async for`. Same runner, same
generator, same breakpoints — minus HTTP, SSE, and the proxy.

    uv run python -m app.debug_runner

To step through it: set a breakpoint inside ocean_runner (e.g. the `yield Token(...)`
line in app/chat_app.py) — OceanRunner.__call__ delegates to it, so it still fires —
and run THIS file under the PyCharm/VS Code debugger.
"""
import asyncio

from agent_chat.protocols import TurnRequest

from app.chat_app import OceanRunner  # same runner class the server uses
from app.context import build_context, AppContext

# Edit this to ask different things. The [query photo at: <path>] marker is what makes
# the agent call photo_id; drop it and you'll get the no-tools answer instead.
QUESTION = (
    "Is this whale at risk from ship strikes?\n\n"
    "[query photo at: /Users/chrislomeli/Source/DATA/oceans/images/556/60028.jpg]"
)


async def main(ctx: AppContext) -> None:
    # Same OceanRunner the server builds — here we hand it ctx directly instead of
    # through a lifespan. Breakpoints in ocean_runner still fire (__call__ delegates).
    runner = OceanRunner(ctx)

    request = TurnRequest(session_id="123", message=QUESTION, metadata={})

    # Each loop iteration receives ONE frame the runner yielded — the exact frames the
    # server would SSE-encode to the browser. Print the type + content so the stream
    # is readable as it goes.
    async for frame in runner(request):
        print(f"{type(frame).__name__:10} {frame}")


if __name__ == "__main__":
    ctx: AppContext = build_context()
    asyncio.run(main(ctx))
