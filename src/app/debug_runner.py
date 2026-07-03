"""debug_runner.py — drive ocean_runner directly, no server, so you can breakpoint it.

The chat backend (uvicorn) is just ONE caller of ocean_runner. This file is a second
caller: it builds a TurnRequest by hand and pulls frames out of the runner with an
`async for`. Same generator, same breakpoints — minus HTTP, SSE, and the proxy.

    uv run python -m app.debug_runner

To step through it: set a breakpoint inside ocean_runner (e.g. the `yield Token(...)`
line in app/chat_app.py) and run THIS file under the PyCharm/VS Code debugger.
"""
import asyncio

from agent_chat.protocols import TurnRequest

from app.chat_app import ocean_runner  # your callback — imported and called directly
from config import get_settings

# Edit this to ask different things. The [query photo at: <path>] marker is what makes
# the agent call photo_id; drop it and you'll get the no-tools answer instead.
QUESTION = (
    "Is this whale at risk from ship strikes?\n\n"
    "[query photo at: /Users/chrislomeli/Source/DATA/oceans/images/556/60028.jpg]"
)


async def main() -> None:

    request = TurnRequest(session_id="123", message=QUESTION, metadata={})

    # Each loop iteration receives ONE frame the runner yielded — the exact frames the
    # server would SSE-encode to the browser. Print the type + content so the stream
    # is readable as it goes.
    async for frame in ocean_runner(request):
        print(f"{type(frame).__name__:10} {frame}")


if __name__ == "__main__":
    get_settings()
    asyncio.run(main())
