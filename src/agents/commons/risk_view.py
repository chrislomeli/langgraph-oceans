"""agents.commons.risk_view

Read seam for per-cell fire-risk scores.

The asymmetry this closes
─────────────────────────
Risk is *written* through a seam: ``report_risk`` persists to the optional
``BaseStore`` and ``evaluate`` writes ``CellRiskAssessment`` onto the world
grid. But risk was *read* by scanning ``world_engine.grid`` inline inside
logistics ``sector_analysis``. That meant binding a persistent risk store
(needed by a stateless / Bedrock-style deployment where the orchestrator
keeps no in-memory world across invocations) could never actually feed
hotspot discovery — the writer had a seam, the reader did not.

``RiskView`` makes the read symmetric with the write. Hotspot discovery
goes through this interface. The in-memory ``GridRiskView`` binding scans
the grid exactly as the old inline loop did (zero behaviour change — the
one-shot demo is unaffected). A future persistent binding implements the
same Protocol with no change to ``sector_analysis``.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable


class RiskHotspot(NamedTuple):
    row: int
    col: int
    risk_score: int
    confidence: int


@runtime_checkable
class RiskView(Protocol):
    """Read-only view over per-cell risk, queried by hotspot discovery."""

    def hotspots(self, min_score: int) -> list[RiskHotspot]:
        """Cells with risk_score >= min_score, in row-major order."""
        ...


class GridRiskView:
    """RiskView bound to the in-memory world grid.

    Reproduces the exact scan logistics used inline: row-major iteration,
    only cells carrying a ``CellRiskAssessment``, filtered by ``min_score``.
    This is the local/demo binding — identical results, now behind the seam.
    """

    def __init__(self, grid) -> None:
        self._grid = grid

    def hotspots(self, min_score: int) -> list[RiskHotspot]:
        from agents.commons.schemas import CellRiskAssessment

        found: list[RiskHotspot] = []
        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                cell = self._grid.get_cell(row, col)
                assessment = cell.risk_assessment
                if not isinstance(assessment, CellRiskAssessment):
                    continue
                if assessment.risk_score >= min_score:
                    found.append(
                        RiskHotspot(
                            row, col, assessment.risk_score, assessment.confidence
                        )
                    )
        return found
