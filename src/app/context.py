"""app/context.py — the composition root (process-lifecycle STARTUP phase).

    import   → definitions only, no heavy work
    startup  → build_context() runs ONCE, returns the ConversationService     ← here
    request  → callers invoke service methods; nothing app-scoped is built
    shutdown → teardown (close pools) — later, in lifespan after `yield`

build_context() is the BUILDER: it assembles the capabilities (settings, registries,
store) plus the process-scoped saver, then constructs the ONE ConversationService the
process shares. It does NOT define or hold the application logic — the graph lives
inside the service (application), built from the capabilities (framework). Keep this
root thin: wire ingredients, hand back the assembled product.

Both entry points build the same way:
  - server  (ocean_runner): lifespan calls build_context() at startup
  - debug   (debug_driver):  main() calls build_context() directly — no server
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.conversation_service import ConversationService
from core.agents.dependencies import AgentDependencies
from core.config import get_settings
from core.llm.llm_registry import build_llm_registry, LLMLabel
from core.prompts import PromptRegistry
from stores.postgres import get_pg_gateway


def build_context() -> ConversationService:
    """Assemble the process-scoped ConversationService. Runs once, at startup."""
    # settings + LangSmith tracing
    settings = get_settings()
    settings.apply_langsmith()

    # capabilities bag (framework/infrastructure — the ingredients the service consumes)
    data_store = get_pg_gateway()
    llm_registry = build_llm_registry(
        settings=settings,     # deployment config (keys, env)
        role_config={          # the roles THIS app wants, and the model behind each
            "oceans_agent": LLMLabel.OPUS,
        })                     # model_catalog defaults to the registry's own `models`
    prompt_registry = PromptRegistry()
    caps = AgentDependencies(
        prompt_registry=prompt_registry,
        llm_registry=llm_registry,
        data_store=data_store,
    )

    # process-scoped checkpointer. B1: MemorySaver (in-proc). B1.5 swaps in
    # AsyncPostgresSaver + a shutdown teardown story. The service builds its graph
    # from caps + saver.
    saver = MemorySaver()

    # the application surface — the one object the process shares across all requests
    return ConversationService(caps, saver=saver)
