"""contracts.py — Layer 3: the shared tool contracts (one interface the agent learns once).

Every tool speaks this language: it accepts the SAME `Filters` shape (ignoring
keys that don't apply) and returns a `ToolResult` (or a subclass). The agent
learns one contract, not five.

The load-bearing rule lives here: a tool is "dumb" — it surfaces signals
(uncertainty, derived geometry, provenance) and NEVER makes a control-flow
decision or raises into the agent loop. On empty/failure it returns
`ToolResult(ok=False)` so the agent can *observe* the failure and recover. All
agency lives one layer up (Layer 2, the agent graph).
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """A geographic bounding box — the geo lingua franca shared across tools."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class Filters(BaseModel):
    """The SAME shape every tool accepts; a tool ignores keys it can't use.

    One `Filters` → one WHERE-builder, reused across tables. The image table
    (`fluke_embeddings`) inherits species/region/date by JOINING through
    individuals/sightings, not by carrying those columns itself.
    """

    species: list[str] | None = None
    region: BBox | None = None
    date_range: tuple[date, date] | None = None
    source: list[str] | None = None  # e.g. ["NOAA_stock_assessment","OBIS","sighting_reports"]


class Citation(BaseModel):
    """Provenance for one piece of evidence — every returned item carries one."""

    kind: Literal["document", "sighting", "catalog_image", "ais"]
    source: str
    locator: str | None = None  # e.g. blob path, page anchor, raster cell
    ref: str | None = None  # e.g. the individual/document id the item belongs to


class ToolResult(BaseModel):
    """The base every tool return extends.

    `ok=False` (never an exception) is how a tool reports empty/failure; `summary`
    is one line the agent + the decision trace can read.
    """

    tool: str
    ok: bool
    summary: str
    citations: list[Citation] = Field(default_factory=list)
