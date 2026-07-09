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

from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.conversation_service import ConversationService
from core.agents.dependencies import AgentDependencies
from core.config import get_settings
from core.context import ContextManager, ContextPolicy
from core.context.summarizer import LLMSummarizer
from core.llm.llm_registry import build_llm_registry, LLMLabel
from core.llm.token_counter import HeuristicTokenCounter
from core.prompts import PromptRegistry
from stores.checkpointer import make_postgres_checkpointer
from stores.postgres import get_pg_gateway


@asynccontextmanager
async def build_context() -> AsyncIterator[ConversationService]:
    """Assemble the process-scoped ConversationService. Runs once, at startup."""
    # settings + LangSmith tracing
    settings = get_settings()
    settings.apply_langsmith()

    # capabilities bag (framework/infrastructure — the ingredients the service consumes)
    postgres_gateway = get_pg_gateway()
    llm_registry = build_llm_registry(
        settings=settings,  # deployment config (keys, env)
        role_config={  # the roles THIS app wants, and the model behind each
            "oceans_agent": LLMLabel.OPUS,
            "summarizer": LLMLabel.HAIKU,
        })  # model_catalog defaults to the registry's own `models`

    prompt_registry = PromptRegistry()

    context_manager = ContextManager(
        counter=HeuristicTokenCounter(),  # core/llm — the salvaged counter
        policies={
            "oceans_agent": ContextPolicy(
                chunk_token_budget=2000,  # size of one summarized chunk
                live_tail_size=6,  # trailing messages never chunked / never compacted
                compaction_size_chars=1000  # tool results smaller than this are left full
            )},
    )
    # stash on AppContext.deps so the graph builder can reach it


    caps = AgentDependencies(
        prompt_registry=prompt_registry,
        llm_registry=llm_registry,
        data_store=postgres_gateway,
        context_manager=context_manager
    )


    # process-scoped checkpointer (AsyncPostgresSaver). Its `async with` holds the
    # connection pool open for the life of this block, so the pool is torn down only
    # when the caller's `async with build_context()` exits. The service builds its
    # graph from caps + saver.
    async with make_postgres_checkpointer(settings.postgres_url, setup=True) as saver:
        yield ConversationService(caps, saver=saver)


