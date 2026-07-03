"""
agents.commons

Public API surface for the generic agent-graph infrastructure (ported from the
world-simulator project). Import from here rather than from submodules to stay
resilient to internal reorganization.

Quick reference:
  - StatusValue   → state machine enum (idle, processing, completed, error)
  - TracedState   → base state contract (session_id, status, error) node_executor needs
  - NodeError     → structured error record captured on a node exception
  - node_executor → decorator: metrics + error handling around a graph node
  - route_base    → generic conditional-edge router (currently unused by the ocean graph)
"""

from agents.commons.node_executor import node_executor
from agents.commons.node_types import NodeError, TracedState
from agents.commons.routing import route_base
from agents.commons.state_types import StatusValue

__all__ = [
    "StatusValue",
    "TracedState",
    "NodeError",
    "node_executor",
    "route_base",
]
