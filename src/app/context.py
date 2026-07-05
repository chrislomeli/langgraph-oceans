"""app/context.py — the composition root.

The phase-2 (STARTUP) function of the app lifecycle:

    import   → definitions only, no heavy work
    startup  → build_context() runs ONCE, returns the AppContext for the process   ← here
    request  → the runner receives (request, ctx); builds nothing app-scoped
    shutdown → teardown (close pools) — later, in lifespan after `yield`

`AppContext` is what the process assembles at startup and hands to the runner. It holds the
`graph` (the assembled product). It will grow a `deps: AgentDependencies` field in A1 — the
raw *ingredients* (llm_registry, prompt_registry, store) the builder consumes — per the
Option-1 decision: don't add a field until something reads it.

Both entry points build this the same way:
  - server  (chat_app):   lifespan builds ctx at startup, a closure hands it to ocean_runner
  - debug   (debug_runner): main() calls build_context() directly — no server
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agents.sandbox_agent.graph import build_sandbox_graph
from core.agents.dependencies import AgentDependencies
from core.config import get_settings
from core.llm.llm_registry import build_llm_registry, LLMLabel
from core.prompts import PromptRegistry
from stores.postgres import get_pg_gateway

if TYPE_CHECKING:
    # Under `from __future__ import annotations` these are string hints at runtime, so the
    # imports are type-checker-only — no runtime cost, no version-path fragility.
    from langgraph.graph.state import CompiledStateGraph
    from core.config import Settings


@dataclass
class AppContext:
    """Everything the process builds once at startup and shares across all requests.

    A pure holder — no building logic lives here. `build_context()` assembles the
    ingredients and fills these fields; the runner only reads them.
    """

    settings: Settings
    graph: CompiledStateGraph
    deps: AgentDependencies


def build_context() -> AppContext:
    """Assemble the process-scoped context. Runs once, at startup — never at import."""
    #  load settings, and apply LangSmith tracing (as chat_app does today)
    settings = get_settings()
    settings.apply_langsmith()

    #   2. build the graph the CURRENT (argless) way — leave _llm alone until A1
    graph = build_sandbox_graph()

    # initialize the postgres datastore
    data_store = get_pg_gateway()

    # set up the models we'll be using
    llm_registry = build_llm_registry(
        settings=settings,     # deployment config (keys, env)
        role_config={          # the roles THIS app wants, and the model behind each
            "ocean_agent": LLMLabel.OPUS,
        })                     # model_catalog defaults to the registry's own `models`

    # point to prompt registry
    prompt_registry = PromptRegistry()
    # prompt_registry.register_models()   # models that we want to provide templates for in system prompts

    # create the composite dependencies that we'll inject to graphs
    agent_dependencies = AgentDependencies(
        prompt_registry=prompt_registry,
        llm_registry=llm_registry,
        data_store=data_store,
    )

    #  create the application context
    ctx = AppContext(settings=settings, graph=graph, deps=agent_dependencies)
    return ctx


