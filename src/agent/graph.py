"""agent/graph.py — B5: the agent graph (LangGraph ReAct loop over the bound tools).

The canonical ReAct loop: the LLM sees the question + tool results so far and either
emits a tool call or a final answer. The agency is the LLM choosing which tool to call
and when to stop — no scripted chain. Design: docs/design/agent-graph-design.md.

    START → agent ──(tool_calls?)──► tools ──► agent ──► … ──(none)──► END

═══════════════════════════════════════════════════════════════════════════════════
TIER SPLIT (memory: agentic-division-of-labor)
  • Claude scaffolded the wiring below (state, model+tools binding, the ReAct loop).
  • TODO(you) tier 2: write SYSTEM_PROMPT — the agent's BRAIN. Plus the tool ads in
    agent/tools.py. Those two are where the agent's behavior lives; everything else is
    plumbing you can read once.
═══════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import TOOLS
from config import get_settings

# DECIDE (tier 3): claude-opus-4-8 = strongest reasoning for decomposition;
# claude-sonnet-4-6 = cheaper/faster if the loop iterates a lot. One-line swap.
MODEL = "claude-opus-4-8"


# ── TODO(you) tier 2: the agent's brain ────────────────────────────────────────────
# This must cover, at minimum:
#   • SCOPE — a conservation-risk reasoning agent (identity·location·conditions·
#     population·threats·protection·ship-risk), NOT a general whale Q&A bot.
#   • DECOMPOSE — the hub-and-zoom reflex: anchor on the individual, then zoom to its
#     life / its kind / its waters; headline answers are CHAINS, not single lookups
#     (see agent-orchestration-design.md).
#   • ABSTAIN — say "not in my sources" outside the lane; treat photo_id NOVEL/abstain
#     as a SOFT prior, corroborate with margin + sighting/location, never a verdict.
#   • CITE — ground claims in what the tools returned.
SYSTEM_PROMPT = "STUB — replace. You are a whale assistant."  # ← TODO(you): write the real prompt


def _llm():
    """ChatAnthropic bound to the tools (so the model can emit tool calls)."""
    s = get_settings()
    key = s.anthropic_api_key.get_secret_value() if s.anthropic_api_key else None
    return ChatAnthropic(model=MODEL, api_key=key, temperature=0).bind_tools(TOOLS)


def build_graph():
    """Compile the ReAct graph. Constructing the LLM is cheap (no API call until invoke)."""
    llm = _llm()

    def agent_node(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    b = StateGraph(MessagesState)
    b.add_node("agent", agent_node)
    b.add_node("tools", ToolNode(TOOLS))
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", tools_condition)  # → tools if tool_calls else END
    b.add_edge("tools", "agent")
    return b.compile()


def run_agent(question: str, image_path: str | None = None) -> list:
    """One question (+ optional photo path) → the full message list (answer + trace).

    The agent does NOT see the image — it gets the path as text and hands it to the
    photo_id tool, which does the embedding. Returns every message (the decision trace);
    the final message is the answer.
    """
    content = question if not image_path else f"{question}\n\n[query photo at: {image_path}]"
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage(content)]})
    return result["messages"]


if __name__ == "__main__":
    # Smoke: compile the graph and show its structure. No LLM call (no cost).
    # Run: uv run python -m agent.graph
    g = build_graph()
    print("compiled OK. nodes:", list(g.get_graph().nodes))
    print("tools bound:", [t.name for t in TOOLS])
    print("\nSYSTEM_PROMPT is a stub:", SYSTEM_PROMPT.startswith("STUB"))
