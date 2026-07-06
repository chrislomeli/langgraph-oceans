"""agents/graph.py — B5: the agents graph (LangGraph ReAct loop over the bound tools).

The canonical ReAct loop: the LLM sees the question + tool results so far and either
emits a tool call or a final answer. The agency is the LLM choosing which tool to call
and when to stop — no scripted chain. Design: docs/design/agents-graph-design.md.

    START → agents ──(tool_calls?)──► tools ──► agents ──► … ──(none)──► END

═══════════════════════════════════════════════════════════════════════════════════
TIER SPLIT (memory: agentic-division-of-labor)
  • Claude scaffolded the wiring below (state, model+tools binding, the ReAct loop).
    agents/tools.py. Those two are where the agents's behavior lives; everything else is
    plumbing you can read once.
═══════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import Field

from agents.tools import TOOLS
from core.agents import node_executor
from core.agents.dependencies import AgentDependencies
from core.agents.node_types import TracedState
from core.config import get_settings
from core.llm.llm_registry import build_llm_registry, LLMLabel
from core.prompts import PromptRegistry
from stores.postgres import get_pg_gateway


class OceanState(TracedState):
    """The agent's graph state.

    Inherits the traced infrastructure fields (session_id, status, error) from
    TracedState — so `node_executor`'s metrics/error handling actually have a home —
    and adds the ReAct message channel with the `add_messages` reducer (the same
    appending behavior LangGraph's MessagesState gives, now on a pydantic base).
    """
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list) # noqa


def make_agent_node(tools: list, deps: AgentDependencies):
    role = "oceans_agent"
    system_prompt =  deps.prompt_registry.render(role, {})
    llm_base = deps.llm_registry.get(role)
    llm = llm_base.bind_tools(tools)


    @node_executor("agent_node")
    def agent_node(state: OceanState) -> dict:
        messages = [SystemMessage(system_prompt)] + state.messages
        return {"messages": [llm.invoke(messages)]}
    return agent_node

def build_sandbox_graph(deps: AgentDependencies):
    """Compile the ReAct graph. Constructing the LLM is cheap (no API call until invoke)."""

    b = StateGraph(OceanState)  # noqa  (PyCharm can't match TypedDict to StateT bound)
    b.add_node("agents", make_agent_node(tools=TOOLS, deps=deps))  # noqa  (PyCharm can't match TypedDict to StateT bound)
    b.add_node("tools", ToolNode(TOOLS)) # all tools as a default

    b.add_edge(START, "agents")
    b.add_conditional_edges("agents", tools_condition)  # → tools if tool_calls else END
    b.add_edge("tools", "agents")
    return b.compile()



if __name__ == '__main__':
    settings = get_settings()

    # initialize the postgres datastore
    data_store = get_pg_gateway()

    # set up the models we'll be using
    llm_registry = build_llm_registry(
        settings=settings,     # deployment config (keys, env)
        role_config={          # the roles THIS app wants, and the model behind each
            "oceans_agent": LLMLabel.OPUS,
        })                     # model_catalog defaults to the registry's own `models`

    # point to prompt registry
    prompt_registry = PromptRegistry()

    # create the composite dependencies that we'll inject to graphs
    agent_dependencies = AgentDependencies(
        prompt_registry=prompt_registry,
        llm_registry=llm_registry,
        data_store=data_store,
    )

    graph = build_sandbox_graph(agent_dependencies)
    print(graph.get_graph().draw_ascii())




