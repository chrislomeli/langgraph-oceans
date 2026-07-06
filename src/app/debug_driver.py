"""debug_driver.py — drive the OceanRunner directly, no server, so you can breakpoint it.

The chat backend (uvicorn) is one caller of OceanRunner: it binds ctx via a lifespan.
This file is a second caller: it hands ctx straight to OceanRunner(ctx), builds a
TurnRequest by hand, and pulls frames out with an `async for`. Same runner, same
generator, same breakpoints — minus HTTP, SSE, and the proxy.

    uv run python -m app.debug_driver

To step through it: set a breakpoint inside ocean_runner (e.g. the `yield Token(...)`
line in app/ocean_runner.py) — OceanRunner.__call__ delegates to it, so it still fires —
and run THIS file under the PyCharm/VS Code debugger.
"""
import asyncio
from uuid import uuid4

from agent_chat.protocols import TurnRequest

from app.ocean_runner import OceanRunner  # same runner class the server uses
from app.context import build_context, AppContext

# Edit this to ask different things. The [query photo at: <path>] marker is what makes
# the agent call photo_id; drop it and you'll get the no-tools answer instead.
QUESTION = (
    "Is this whale at risk from ship strikes?\n\n"
    "[query photo at: /Users/chrislomeli/Source/DATA/oceans/images/556/60028.jpg]"
)

def get_question():
    yield QUESTION


async def ask_ai(ocean_runner:  OceanRunner):
    session_id = str(uuid4())

    q = next(get_question())
    request = TurnRequest(session_id=session_id, message=q, metadata={})

    async for frame in ocean_runner(request):
        print(f"{type(frame).__name__:10} {frame}")


async def main() -> None:
    # Same OceanRunner the server builds — here we hand it ctx directly instead of
    # through a lifespan. Breakpoints in ocean_runner still fire (__call__ delegates).
    runner = OceanRunner(build_context())

    request = TurnRequest(session_id="123", message=QUESTION, metadata={})

    # Each loop iteration receives ONE frame the runner yielded — the exact frames the
    # server would SSE-encode to the browser. Print the type + content so the stream
    # is readable as it goes.
    async for frame in runner(request):
        print(f"{type(frame).__name__:10} {frame}")


if __name__ == "__main__":
    asyncio.run(main())
