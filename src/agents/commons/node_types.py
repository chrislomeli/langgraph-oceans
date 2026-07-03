"""
world-simulator.agents.commons.node_types

Node infrastructure primitives: error records and the base state contract
that ``node_executor`` requires from every graph state.

Kept separate from domain schemas (``wildfire_schemas.py``) so that the executor
and router can import these without pulling in LLMRegistry, PromptRegistry,
or any other heavy dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from agents.commons.state_types import StatusValue


class NodeError(BaseModel):
    """
    Structured error record for exceptions caught by node_executor.

    Captures everything needed for distributed debugging: the original
    exception message, full traceback, which node failed, and which
    session/request was being processed. Written to state.error when
    a node raises, then surfaced in logs and (if configured) tracing.

    Fields:
        message    : str(exception) — human-readable error description
        traceback  : Full stack trace for debugging
        node       : Logical node name from node_executor decorator
        session_id : Request correlation ID for tracing across nodes
        timestamp  : UTC time when error was captured
    """

    message: str = Field(description="Exception message or str(exception)")
    traceback: str = Field(description="Full stack trace for debugging")
    node: str = Field(description="Logical node name from @node_executor")
    session_id: str = Field(description="Request correlation ID for distributed tracing")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when error was captured",
    )


class TracedState(BaseModel):
    """
    Minimum contract that node_executor requires from any graph state.

    Agent state classes inherit from this to get the three fields the
    execution framework needs:
      - session_id: for request tracing across nodes
      - status: for the state machine (idle/processing/completed/error)
      - error: for structured error capture on exception
    """

    session_id: str | None = Field(
        default=None, description="Request correlation ID for tracing across nodes/graphs"
    )
    status: StatusValue = Field(default=StatusValue.IDLE, description="Current state machine value")
    error: NodeError | None = Field(
        default=None, description="Structured error record if last node raised exception"
    )
