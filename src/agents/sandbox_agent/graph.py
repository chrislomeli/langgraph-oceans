"""agents/graph.py — B5: the agents graph (LangGraph ReAct loop over the bound tools).

The canonical ReAct loop: the LLM sees the question + tool results so far and either
emits a tool call or a final answer. The agency is the LLM choosing which tool to call
and when to stop — no scripted chain. Design: docs/design/agents-graph-design.md.

    START → agents ──(tool_calls?)──► tools ──► agents ──► … ──(none)──► END

═══════════════════════════════════════════════════════════════════════════════════
TIER SPLIT (memory: agentic-division-of-labor)
  • Claude scaffolded the wiring below (state, model+tools binding, the ReAct loop).
  • TODO(you) tier 2: write SYSTEM_PROMPT — the agents's BRAIN. Plus the tool ads in
    agents/tools.py. Those two are where the agents's behavior lives; everything else is
    plumbing you can read once.
═══════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import Field

from agents.commons import node_executor
from agents.commons.node_types import TracedState
from agents.tools import TOOLS
from config import get_settings


class OceanState(TracedState):
    """The agent's graph state.

    Inherits the traced infrastructure fields (session_id, status, error) from
    TracedState — so `node_executor`'s metrics/error handling actually have a home —
    and adds the ReAct message channel with the `add_messages` reducer (the same
    appending behavior LangGraph's MessagesState gives, now on a pydantic base).
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


# DECIDE (tier 3): claude-opus-4-8 = strongest reasoning for decomposition;
# claude-sonnet-4-6 = cheaper/faster if the loop iterates a lot. One-line swap.
MODEL = "claude-opus-4-8"


# Starter written by Claude 2026-06-21 (per agentic-division-of-labor: starters to
# learn from + tune against real traces). Tune this against what the agents actually does.
SYSTEM_PROMPT = """You are a marine-mammal CONSERVATION-RISK assistant. You reason about an \
individual whale's identity, where it ranges, the conditions of its habitat, and its exposure \
to ship strikes — and you say so plainly when a question falls outside that lane.

WHAT YOU COVER: identity (who is this whale), location/range (where it's been seen), habitat \
conditions (depth, temperature, sanctuary), and ship-strike exposure (vessel traffic over its \
range, protected zones).
WHAT YOU DON'T: health, age, family, individual behavior, or real-time positions. For those, \
say "that's not in my sources" rather than guessing.

HOW TO ANSWER:
- Anchor on the individual, then gather what you need by CHAINING tools — most real answers are \
a chain, not one lookup. A typical ship-risk question goes: identify the whale (photo_id) -> \
find its range (sighting_lookup) -> measure ship traffic over that range (vessel_traffic), \
enriching with habitat conditions (sighting_context) where it sharpens the picture.
- Thread one tool's output into the next: the individual_id from photo_id drives the sighting \
tools; the range box from sighting_lookup drives vessel_traffic.
- Decide for yourself which tools a question needs — not every question needs all of them.

UNCERTAINTY: the photo_id NOVEL/abstain signal is a SOFT prior, not a verdict — corroborate it \
with the match margin and sighting/location evidence before concluding a whale is new or \
misidentified.

GROUNDING: base every factual claim on what the tools returned, and attribute it (which whale, \
which data). If the tools don't support an answer, say so."""


def _llm():
    """ChatAnthropic bound to the tools (so the model can emit tool calls)."""
    s = get_settings()
    key = s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else None
    # NB: claude-opus-4-8 deprecates `temperature` (the API rejects it); omit it.
    return ChatAnthropic(model=MODEL, api_key=key).bind_tools(TOOLS)


def build_graph():
    """Compile the ReAct graph. Constructing the LLM is cheap (no API call until invoke)."""
    llm = _llm()

    @node_executor("agent_node")
    def agent_node(state: OceanState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT)] + state.messages
        return {"messages": [llm.invoke(messages)]}


    b = StateGraph(OceanState)  # noqa  (PyCharm can't match TypedDict to StateT bound)
    b.add_node("agents", agent_node)  # noqa  (PyCharm can't match TypedDict to StateT bound)
    b.add_node("tools", ToolNode(TOOLS))
    b.add_edge(START, "agents")
    b.add_conditional_edges("agents", tools_condition)  # → tools if tool_calls else END
    b.add_edge("tools", "agents")
    return b.compile()


def run_agent(question: str, image_path: str | None = None) -> list:
    """One question (+ optional photo path) → the full message list (answer + trace).

    The agents does NOT see the image — it gets the path as text and hands it to the
    photo_id tool, which does the embedding. Returns every message (the decision trace);
    the final message is the answer.
    """
    content = question if not image_path else f"{question}\n\n[query photo at: {image_path}]"
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage(content)]})  # noqa  (PyCharm can't match TypedDict to StateT bound) 
    return result["messages"]


if __name__ == "__main__":
    # Smoke: compile the graph and show its structure. No LLM call (no cost).
    # Run: uv run python -m agents.graph
    g = build_graph()
    print("compiled OK. nodes:", list(g.get_graph().nodes))
    print("tools bound:", [t.name for t in TOOLS])
    print("\nSYSTEM_PROMPT is a stub:", SYSTEM_PROMPT.startswith("STUB"))
