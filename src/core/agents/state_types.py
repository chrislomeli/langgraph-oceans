"""
world-simulator.agents.state_types

Shared state primitives used by both the cluster agents and the supervisor.

Keeping these here prevents the supervisor from importing from the cluster
module (or vice versa) just to get a shared enum.
"""

from enum import StrEnum


class StatusValue(StrEnum):
    """
    State machine values for agents workflow status.

    Every node in a graph reads and writes this enum to coordinate
    execution flow. The node_executor decorator uses this to decide
    whether to surface errors to the state and route_base uses it
    to determine the next node in conditional edges.

    Values:
        IDLE        → Initial state, waiting for work
        PROCESSING  → Node is actively working
        COMPLETED   → Workflow finished successfully
        ERROR       → Exception caught, error recorded in state.error
    """

    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
