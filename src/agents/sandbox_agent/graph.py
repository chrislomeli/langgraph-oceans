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

from typing import Optional, Literal, Any

from langchain_core.messages import SystemMessage, AnyMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from agents.sandbox_agent.ocean_state import OceanState
from agents.tools import TOOLS
from core.agents import node_executor
from core.agents.dependencies import AgentDependencies
from core.config import get_settings
from core.context import ContextManager
from core.context.summarizer import LLMSummarizer
from core.llm.llm_registry import build_llm_registry, LLMLabel
from core.llm.token_counter import HeuristicTokenCounter
from core.prompts import PromptRegistry
from stores.postgres import get_pg_gateway




def make_summarizer_node(deps: AgentDependencies):
    """
    Update the state's summary_chunks and last message id
     Call the context framework to perform summary and return partials
     summary_chunks and last message id both come from te mixin

    :param deps:
    :return:
    """
    role = "summarizer"
    system_prompt = deps.prompt_registry.render(role, {})
    llm_summarizer = deps.llm_registry.get(role)
    summarizer = LLMSummarizer(llm_summarizer, system_prompt)
    context_manager = deps.context_manager

    @node_executor("summarizer_node")
    def summarizer_node(state: OceanState):
        # state has summaries, messages, last_ id -- summarize and update them here before calling the agent
        context_manager.prepare(state=state, policy_key=role, summarizer=summarizer)

        last_item =  state.messages[-1]
        if isinstance(last_item, ToolMessage):
            pass


    return summarizer_node


def make_agent_node(tools: list, deps: AgentDependencies):
    role = "oceans_agent"
    llm_agent = deps.llm_registry.get(role)
    system_prompt = deps.prompt_registry.render(role, {})
    context = deps.context_manager

    @node_executor("agent_node")
    def agent_node(state: OceanState) -> dict:

        messages_view = context.build_view(state, "oceans_agent")
        messages = [SystemMessage(system_prompt)] + messages_view

        return {"messages": [llm_agent.invoke(messages)]}

    return agent_node


def route_ai_response(state: list[AnyMessage] | dict[str, Any] | BaseModel, messages_key: str = "messages",) -> Literal["tools", "__end__"]:
    # tools_condition routes to "tools" or "__end__"
    next_node =  tools_condition(state,messages_key)

    # perform any additional work
    return next_node




def build_sandbox_graph(deps: AgentDependencies, saver: Optional[BaseCheckpointSaver] = None):
    """Compile the ReAct graph. Constructing the LLM is cheap (no API call until invoke)."""

    b = StateGraph(OceanState)  # noqa  (PyCharm can't match TypedDict to StateT bound)
    b.add_node("summarizer",
               make_summarizer_node(deps=deps))  # noqa  (PyCharm can't match TypedDict to StateT bound)
    b.add_node("agents",
               make_agent_node(tools=TOOLS, deps=deps))  # noqa  (PyCharm can't match TypedDict to StateT bound)
    b.add_node("tools", ToolNode(TOOLS))  # all tools as a default

    b.add_edge(START, "summarizer")
    b.add_edge("summarizer", "agents")
    b.add_conditional_edges("agents", route_ai_response)  # → use a hof in case we want to pass anything in
    b.add_edge("tools", "summarizer")
    return b.compile(saver)


if __name__ == '__main__':
    settings = get_settings()

    # initialize the postgres datastore
    data_store = get_pg_gateway()

    # set up the models we'll be using
    llm_registry = build_llm_registry(
        settings=settings,  # deployment config (keys, env)
        role_config={  # the roles THIS app wants, and the model behind each
            "oceans_agent": LLMLabel.OPUS,
        })  # model_catalog defaults to the registry's own `models`

    # point to prompt registry
    prompt_registry = PromptRegistry()

    # create the composite dependencies that we'll inject to graphs
    agent_dependencies = AgentDependencies(
        prompt_registry=prompt_registry,
        llm_registry=llm_registry,
        data_store=data_store,
        context_manager=ContextManager(counter=HeuristicTokenCounter())
    )

    # saver is irrelevant to draw_ascii (structure only), so omit it — the arg is
    # optional and defaults to None.
    graph = build_sandbox_graph(agent_dependencies)
    print(graph.get_graph().draw_ascii())
