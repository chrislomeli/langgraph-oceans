"""agent/cli.py — B-CLI: run the agent from the terminal (the vision's "Interface").

    uv run python -m agent.cli "Is this whale at risk from ships?" --image /path/to/fluke.jpg
    uv run python -m agent.cli "What threatens humpbacks on the US West Coast?"
    uv run python -m agent.cli "..." --trace      # also print the decision trace

Tier 1 (Claude-built, read once): arg-parsing + invoke + render. The agent's behavior
lives in agent/graph.py SYSTEM_PROMPT and agent/tools.py ads — not here.
"""

from __future__ import annotations

import argparse

from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import run_agent
from config import get_settings


def _render_trace(messages: list) -> None:
    """Show the decision trace: each tool the agent called + what came back."""
    print("\n── decision trace ──")
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  → call {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            first_line = str(m.content).splitlines()[0] if m.content else ""
            print(f"      {m.name} → {first_line[:100]}")
    print("────────────────────")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the ocean-mammal conservation agent.")
    ap.add_argument("question", help="the question to ask")
    ap.add_argument("--image", default=None, help="path/URL to a query whale-fluke photo")
    ap.add_argument("--trace", action="store_true", help="print the tool-call decision trace")
    args = ap.parse_args()

    get_settings().apply_langsmith()  # optional LangSmith tracing if configured

    messages = run_agent(args.question, args.image)

    if args.trace:
        _render_trace(messages)

    print("\n── answer ──")
    print(messages[-1].content)


if __name__ == "__main__":
    main()
