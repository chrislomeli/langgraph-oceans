"""
world-simulator.agents.commons.deps

Dependency injection container for agents graphs.

Lives in its own module so that domain schemas (``wildfire_schemas.py``) and node
infrastructure (``node_types.py``) remain free of heavy framework imports.
"""

from __future__ import annotations

from typing import Optional

from langgraph.store.base import BaseStore
from pydantic import BaseModel

from core.llm.llm_registry import LLMRegistry
from core.prompts import PromptRegistry
from stores.postgres import PgGateway


class AgentDependencies(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    llm_registry: LLMRegistry
    prompt_registry: PromptRegistry
    data_store: PgGateway
    store: Optional[BaseStore] = None
