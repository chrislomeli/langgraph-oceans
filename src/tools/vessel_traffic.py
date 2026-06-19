"""vessel_traffic.py — Layer 3 tool: "how much ship traffic is over this water?"

The multi-hop payoff tool. `sighting_lookup` computes an individual's range as a
BBox; this tool reads the AIS shipping-traffic raster over that BBox and returns
the traffic metrics. The agent (Layer 2) decides whether that's a *risk* — this
tool only surfaces the numbers (dumb-tool contract, see contracts.py).

The data: `datasets.ais_YYYY` are PostGIS rasters of vessel **transit counts** per
~100 m cell, one table per year, native **EPSG:3857**. A cell's value is how many
vessel transits crossed it that year (a ferry crossing 200× counts 200 — so this
is TRANSITS, not distinct vessels; we name the fields accordingly, unlike the
vision's `vessel_count`).

The query (verified 2026-06-18 on the live DB — SB Channel mean 28.2 transits/cell
vs quiet open ocean 1.2, a 23× contrast):
  1. build the query envelope in 4326, transform to the raster's 3857
  2. clip every intersecting raster tile to it, union into one raster
  3. ST_SummaryStats → count / sum / mean / max over the cells that carry traffic
  4. separately, intersect the range with the NMS/VSR speed-zone polygons →
     `lane_overlap` (is the range inside a managed ship-strike zone?)

Honest semantics (the two modelling choices flagged in design):
  - `mean_transits_per_cell` averages over TRAFFICKED cells only (nodata=0 cells
    are excluded by ST_SummaryStats). It reads "how heavy is traffic where there
    IS traffic," not "averaged over empty ocean." `total_transits` is the raw Σ.
  - Per-YEAR table, never the `ais_all` union: overlapping yearly footprints would
    multi-count the same water.
"""

import logging

from tools.contracts import BBox, Citation, ToolResult
from stores.postgres import get_pg_gateway

log = logging.getLogger(__name__)

# One raster table per year; table names can't be bound as %s params, so we pick
# from this allowlist (never interpolate a caller-supplied year into the SQL).
AIS_YEARS = (2022, 2023, 2024, 2025)
DEFAULT_YEAR = 2024


class VesselTrafficResult(ToolResult):
    """Traffic metrics over a region — raw signals; the agent judges 'risk'."""

    region: BBox
    year: int
    total_transits: int  # Σ transit counts over the range (the headline aggregate)
    trafficked_cells: int  # how many ~100 m cells carried any traffic
    mean_transits_per_cell: float  # density proxy, over trafficked cells only
    busiest_cell: int  # max transit count in any single cell (the hotspot)
    lane_overlap: bool  # does the range intersect a managed speed-zone polygon?
    overlapping_zones: list[str] = []  # which NMS/VSR zones (provenance)


# Aggregate the AIS raster over a 4326 envelope. {table} is from the allowlist
# above (not caller data); the four coordinates travel as %s params.
_TRAFFIC_SQL = """
    WITH env AS (
        SELECT ST_Transform(ST_MakeEnvelope(%s, %s, %s, %s, 4326), 3857) AS g
    ),
    clipped AS (
        SELECT ST_Union(ST_Clip(r.rast, env.g, true)) AS rast
        FROM datasets.{table} r, env
        WHERE ST_Intersects(r.rast, env.g)        -- convexhull GiST index
    )
    SELECT (ss).count AS trafficked_cells,
           (ss).sum   AS total_transits,
           (ss).mean  AS mean_per_cell,
           (ss).max   AS busiest_cell
    FROM clipped, LATERAL ST_SummaryStats(clipped.rast) ss
"""

# Which managed speed-zone polygons does the range fall in? geom is 4326, so the
# 4326 envelope intersects it directly (no transform).
_ZONE_SQL = """
    SELECT name
    FROM datasets.vsr_zones
    WHERE ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
    ORDER BY name
"""


class VesselTrafficTool:
    """Read ship-traffic metrics for a BBox (or a single point) from the AIS raster."""

    def __init__(self):
        self.gw = get_pg_gateway()
        log.info("VesselTrafficTool ready")

    def query(self, region: BBox, year: int = DEFAULT_YEAR) -> VesselTrafficResult:
        """Traffic metrics over `region` for `year`. Never raises — ok=False if the
        box is outside AIS coverage (no intersecting tiles)."""
        if year not in AIS_YEARS:
            return self._empty(region, year, f"No AIS data for year {year} (have {AIS_YEARS})")

        coords = (region.lon_min, region.lat_min, region.lon_max, region.lat_max)
        sql = _TRAFFIC_SQL.format(table=f"ais_{year}")
        rows = self.gw.fetch_rows(sql, coords)
        row = rows[0] if rows else {}

        # clipped.rast is NULL when no tile intersects → SummaryStats yields NULLs:
        # that's "outside coverage", distinct from "covered water, zero traffic".
        if not row or row.get("total_transits") is None:
            return self._empty(region, year, "Region outside AIS coverage")

        # set() dedups: a sanctuary stored as a multipolygon returns one row per part.
        zones = sorted({r["name"] for r in self.gw.fetch_rows(_ZONE_SQL, coords)})
        total = int(row["total_transits"])
        cells = int(row["trafficked_cells"])
        busiest = int(row["busiest_cell"])
        mean = float(row["mean_per_cell"])

        zone_note = f"; in zone(s): {', '.join(zones)}" if zones else ""
        summary = (
            f"{total:,} vessel transits ({year}) over {cells:,} cells "
            f"(mean {mean:.1f}/cell, busiest {busiest:,}){zone_note}"
        )
        return VesselTrafficResult(
            tool="vessel_traffic",
            ok=True,
            summary=summary,
            citations=[
                Citation(kind="ais", source=f"MarineCadastre transit-counts {year}",
                         locator=f"bbox({coords})", ref=str(year)),
            ],
            region=region,
            year=year,
            total_transits=total,
            trafficked_cells=cells,
            mean_transits_per_cell=mean,
            busiest_cell=busiest,
            lane_overlap=bool(zones),
            overlapping_zones=zones,
        )

    def query_point(self, lat: float, lon: float, year: int = DEFAULT_YEAR) -> VesselTrafficResult:
        """Single-point convenience: a tiny box around (lat, lon). Useful for one
        sighting location rather than a whole range."""
        eps = 0.01  # ~1 km box so the point lands inside a 100 m cell or two
        return self.query(
            BBox(lat_min=lat - eps, lat_max=lat + eps, lon_min=lon - eps, lon_max=lon + eps),
            year=year,
        )

    @staticmethod
    def _empty(region: BBox, year: int, why: str) -> VesselTrafficResult:
        return VesselTrafficResult(
            tool="vessel_traffic", ok=False, summary=why, region=region, year=year,
            total_transits=0, trafficked_cells=0, mean_transits_per_cell=0.0,
            busiest_cell=0, lane_overlap=False, overlapping_zones=[],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tool = VesselTrafficTool()

    # Santa Barbara Channel — a major shipping lane through two sanctuaries.
    sb = BBox(lat_min=33.9, lat_max=34.4, lon_min=-120.5, lon_max=-119.3)
    print("\n--- Santa Barbara Channel (busy) ---")
    r = tool.query(sb)
    print(f"  ok={r.ok}  lane_overlap={r.lane_overlap}  zones={r.overlapping_zones}")
    print(f"  {r.summary}")

    # Quiet open ocean — the contrast that proves the metric discriminates.
    quiet = BBox(lat_min=40.0, lat_max=41.0, lon_min=-130.0, lon_max=-129.0)
    print("\n--- Open ocean (quiet) ---")
    r2 = tool.query(quiet)
    print(f"  ok={r2.ok}  lane_overlap={r2.lane_overlap}")
    print(f"  {r2.summary}")

    if r.ok and r2.ok and r2.total_transits > 0:
        print(f"\n  contrast: SB mean {r.mean_transits_per_cell:.1f} vs quiet "
              f"{r2.mean_transits_per_cell:.1f} per cell "
              f"({r.mean_transits_per_cell / r2.mean_transits_per_cell:.0f}×)")
