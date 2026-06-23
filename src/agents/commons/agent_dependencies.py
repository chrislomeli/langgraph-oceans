"""
world-simulator.agents.commons.deps

Dependency injection container for agents graphs.

Lives in its own module so that domain schemas (``wildfire_schemas.py``) and node
infrastructure (``node_types.py``) remain free of heavy framework imports.
"""

from __future__ import annotations

from langgraph.store.base import BaseStore
from pydantic import BaseModel

from llm.llm_registry import LLMRegistry
from prompts import PromptRegistry


class AgentDependencies(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    llm_registry: LLMRegistry
    prompt_registry: PromptRegistry
    # Optional because every consumer already guards for absence
    # (logistics graph: `data_store is not None`; cluster graph passes
    # cell_state_manager into a `| None` factory). Stub-mode tests build
    # partial deps; production always supplies them via composition.
    store: BaseStore | None = None
