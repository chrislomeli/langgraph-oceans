"""
world-simulator.agents.commons.schemas

Cross-agents data contracts for the risk assessment pipeline.

Design intent
─────────────
These schemas define the data that flows through the cluster agents's
risk pipeline:

    SensorEvent (wire) → CellStateManager → CellReadings → agents → RiskAssessment

CellStateManager maintains a running per-cell snapshot of the latest
metrics, decides when a cell should re-evaluate, writes the values onto
the world grid (session ground truth), and emits a CellReadings envelope
(cluster_id + position) per triggered cell. The cluster agents's
update_world node reads that grid ground truth, then evaluate produces
RiskAssessments and writes them back onto each GenericCell.

Separation of concerns
──────────────────────
  - SensorEvent (transport/wildfire_schemas.py) is the wire format. Domain-agnostic.
  - CellReadings is the agents's input. The minimum needed to update the
    world grid and identify the cell.
  - RiskAssessment is the agents's output. What the supervisor consumes.

Coordinate convention
─────────────────────
GridPosition follows GenericTerrainGrid's convention:
  - row 0 = NORTH edge, increasing row = southward
  - col 0 = WEST edge, increasing col = eastward
  - (0, 0) = north-west corner
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agents.commons.node_types import NodeError
from agents.commons.state_types import StatusValue


class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[32m"
    TEAL = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GREY = "\033[90m"
    RESET = "\033[0m"


# ── Base state ───────────────────────────────────────────────────────────────


class TracedState(BaseModel):
    """
    Minimum contract that node_executor requires from any graph state.

    All agents state classes inherit from this (directly or indirectly)
    to get the three fields that the execution framework needs:
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


# ── Spatial primitives ────────────────────────────────────────────────────────


class GridPosition(BaseModel):
    """A cell's location on the simulation grid.

    Coordinate convention (matches GenericTerrainGrid):
      - row 0 = NORTH edge, increasing row = southward
      - col 0 = WEST edge, increasing col = eastward
      - (0, 0) = north-west corner

    Adjacency: neighbors are ±1 in either axis (8-connected).
    """

    row: int
    col: int
    layer: int = 0  # Default to 0 for 2D scenarios


# ── Sensor reading ────────────────────────────────────────────────────────────


class Metric(BaseModel):
    """A single validated reading extracted from a SensorEvent.

    CellStateManager produces Metrics from raw SensorEvents by:
      1. Extracting the canonical scalar value from the opaque payload
      2. Mapping the sensor's source_id to a GridPosition
      3. Computing signal_strength (sensor confidence × distance decay)
    """

    sensor_id: str = Field(description="Sensor identifier key")
    type: str = Field(
        description="Sensor type: 'temperature', 'humidity', 'wind_speed', 'wind_direction'"
    )
    value: float = Field(description="Canonical scalar value (celsius, %, m/s, degrees)")
    signal_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Combined reliability: sensor confidence × distance decay. "
        "1.0 = sensor is at this cell with full health. "
        "0.0 = reading is unreliable for this cell.",
    )
    source_id: str = Field(description="Which sensor produced this reading")
    position: GridPosition = Field(description="Where the sensor sits on the grid")
    timestamp: datetime = Field(description="When the reading was taken (UTC)")


# ── The agents's input ─────────────────────────────────────────────────────────


class CellReadings(BaseModel):
    """Identifies a single triggered cell (cluster + position).

    The orchestrator groups CellReadings by cluster_id and the supervisor
    fans them out to per-cluster agents. Metric values are written onto
    the world grid upstream by CellStateManager.update(); update_world
    reads that grid ground truth by position — values are no longer
    carried in this envelope.
    """

    cluster_id: str = Field(description="Which cluster this cell belongs to")
    position: GridPosition = Field(description="Grid coordinates of the cell")


# ── The agents's output ────────────────────────────────────────────────────────


class RiskAssessment(BaseModel):
    collated_record_risks: list[CollatedRecordRisk] = Field(
        description="A risk assessment for each cell in the provided cluster",
        default_factory=list,
    )


class CollatedRecordRisk(BaseModel):
    """Fire risk score for an individual cell."""

    position: GridPosition = Field(
        description="a row/column reference to the specific cell we are evaluating"
    )
    risk_score: int = Field(
        ge=0,
        le=10,
        description="Agent's fire danger estimate integer",
    )
    confidence: int = Field(
        ge=0,
        le=3,
        description="confidence in risk_score",
    )
    confidence_rationale: str = Field(
        description="Why the agents chose this confidence level. "
        "e.g. 'Based on 2/3 sensor types with strong signal; "
        "wind data inferred from 6-hour forecast tool.'"
    )
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="What drove the assessment: e.g. ['temp=52°C (>38 threshold)', "
        "'humidity=12% (<15 critical)', 'terrain=grassland (high fuel)']",
    )


class CellRiskAssessment(BaseModel):
    """Minimal fire risk assessment stored on GenericCell.

    Token-efficient subset of CollatedRecordRisk for ground truth storage.
    The rationale is kept for debugging/tracing but omitted from LLM
    context to save tokens (the sector summary provides enough context
    for decisions).
    """

    risk_score: int = Field(ge=0, le=10)
    confidence: int = Field(ge=0, le=3)
    confidence_rationale: str = Field(default="", description="Reasoning for tracing/debug")


# Agent view of a published advisory
class ResourceAdvisory(BaseModel):
    """Structured advisory report for resource deployment decisions."""

    epicenter_row: int = Field(
        description="Terrain grid row index of the fire risk epicenter (highest-risk cell)."
    )

    epicenter_column: int = Field(
        description="Terrain grid column index of the fire risk epicenter (highest-risk cell)."
    )

    location_description: str = Field(
        description="Human-readable description of the affected area, especially for impact zones difficult to describe in grid coordinates (e.g., 'northwest slope below ridgeline')."
    )
    situation: str = Field(
        description="Current fire status, spread direction, and immediate threat level. 1-2 sentences."
    )

    urgency_level: int = Field(
        ge=1,
        le=4,
        description="""How urgent and immediate is this situation?

LEVEL 4 (Fade Out): Lowest readiness; routine monitoring.
LEVEL 3 (Double Take): Elevated readiness; increased monitoring.
LEVEL 2 (Fast Pace): High readiness; prepare for deployment.
LEVEL 1 (Cocked Pistol): Maximum readiness; imminent response required.
""",
    )
    notes: str = Field(
        description=(
            "Context, uncertainties, and edge-case reasoning. "
            "Discuss resource conflicts, conditional scenarios, or cascading risks. "
            "Example: '3 engines committed to 30%-contained Lompoc fire. "
            "If 2+ hotspots ignite simultaneously, Level 1 capacity exceeded.'"
        )
    )
    recommendation: str = Field(
        description="Specific action to take, or 'Monitor only' if no deployment needed."
    )


# Database view of the ResourceAdvisory adds tracking fields
class ResourceAdvisoryRecord(ResourceAdvisory):
    id: UUID = Field(default_factory=uuid4, description="Unique identifier. Generated on creation.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["SENT", "SUPPRESSED", "ACKNOWLEDGED"] = Field(default="SENT")

    def to_db_row(self) -> tuple:
        """Return tuple for INSERT/UPDATE — computed fields excluded."""
        return (
            self.id,
            self.created_at,
            self.status,
            self.epicenter_row,
            self.epicenter_column,
            self.location_description,
            self.situation,
            self.urgency_level,
            self.notes,
            self.recommendation,
        )