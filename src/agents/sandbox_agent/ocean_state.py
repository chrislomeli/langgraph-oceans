from core.agents import TracedState
from core.context.state import ContextStateFields
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages  # same one graph.py used
from pydantic import Field


class OceanState(TracedState, ContextStateFields):
    """The agent's graph state.

    Inherits the traced infrastructure fields (session_id, status, error) from
    TracedState — so `node_executor`'s metrics/error handling actually have a home —
    and adds the ReAct message channel with the `add_messages` reducer (the same
    appending behavior LangGraph's MessagesState gives, now on a pydantic base).
    """
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)  # noqa
