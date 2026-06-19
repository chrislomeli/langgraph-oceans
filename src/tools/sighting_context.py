"""sighting_context.py — Layer 3 tool: "what's the environmental context of this whale's habitat?"

The enrichment we almost went out to acquire (GEBCO bathymetry, ERDDAP SST) turned out to
be **already in the source data**: OBIS-SEAMAP pre-computed a per-sighting `oceano` blob
(depth, SST, salinity, marine ecoregion, sanctuary) for every encounter. It survives in the
staging table `obis_seamap_points.oceano` (100% coverage) and joins to `sightings` via
`source_row_id = row_id`. This tool aggregates it per individual — near-zero cost, no
external API. (Lesson, again: look at the data before importing a playbook.)

Dumb-tool contract: surfaces the numbers; the agent judges. The load-bearing signal is
**depth / shelf-fraction** — the F5 risk *modulation* lever (heavy traffic over a shallow
shelf near approach lanes is strike-relevant; over a deep offshore canyon, less so).

Caveat: this is pre-computed POINT enrichment (one value per sighting), not a raster you can
query for an arbitrary region — so it describes *this whale's sightings*, not arbitrary water.
"""

import logging

from tools.contracts import Citation, ToolResult
from stores.postgres import get_pg_gateway

log = logging.getLogger(__name__)

SHELF_DEPTH_M = -200  # shallower than this = continental shelf (strike-relevant)


class EnvContext(ToolResult):
    """Aggregate environmental context over an individual's sightings."""

    individual_id: int
    n: int = 0
    depth_median_m: float | None = None  # negative = below sea level
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    shelf_fraction: float | None = None  # fraction of sightings over the shelf (<200 m deep)
    sst_median_c: float | None = None
    sst_min_c: float | None = None
    sst_max_c: float | None = None
    regions: list[str] = []  # distinct LMEs (e.g. "California Current")
    sanctuaries: list[str] = []  # distinct WDPA sanctuary names the whale was seen in


# Join sightings → staging, parse the per-sighting oceano JSON, aggregate. The JSON paths
# are fixed literals (no caller data); individual_id travels as a %s param.
_CONTEXT_SQL = """
    WITH s AS (
        SELECT (o.oceano::json->'BATH'->>'ETOPO1')::float           AS depth,
               (o.oceano::json->'SST'->'OISST'->>'DAY')::float      AS sst,
               o.oceano::json->'ZONE'->'LME'->0->>'LME_NAME'        AS lme,
               o.oceano::json->'ZONE'->'WDPA'->0->>'NAME'           AS wdpa
        FROM sightings sg
        JOIN obis_seamap_points o ON o.row_id = sg.source_row_id
        WHERE sg.individual_id = %s
    )
    SELECT count(*)                                                          AS n,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY depth)               AS depth_median,
           min(depth) AS depth_min, max(depth) AS depth_max,
           avg((depth > %s)::int)::float                                    AS shelf_fraction,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY sst)                 AS sst_median,
           min(sst) AS sst_min, max(sst) AS sst_max,
           array_agg(DISTINCT lme)  FILTER (WHERE lme  IS NOT NULL)         AS regions,
           array_agg(DISTINCT wdpa) FILTER (WHERE wdpa IS NOT NULL)         AS sanctuaries
    FROM s
"""


class SightingContextTool:
    """Environmental context for an individual, from the OBIS oceano enrichment."""

    def __init__(self):
        self.gw = get_pg_gateway()
        log.info("SightingContextTool ready")

    def query(self, individual_id: int) -> EnvContext:
        rows = self.gw.fetch_rows(_CONTEXT_SQL, (individual_id, SHELF_DEPTH_M))
        r = rows[0] if rows else {}
        if not r or not r.get("n"):
            return EnvContext(tool="sighting_context", ok=False,
                              summary="No oceano context for this individual", individual_id=individual_id)

        depth_med = r["depth_median"]
        shelf = r["shelf_fraction"]
        sst_med = r["sst_median"]
        regions = r.get("regions") or []
        sanctuaries = r.get("sanctuaries") or []

        bits = [f"{r['n']} sightings"]
        if depth_med is not None:
            bits.append(f"median depth {depth_med:.0f} m"
                        + (f" ({shelf:.0%} over continental shelf)" if shelf is not None else ""))
        if sst_med is not None:
            bits.append(f"SST ~{sst_med:.1f}°C")
        if regions:
            bits.append("regions: " + ", ".join(regions))
        if sanctuaries:
            bits.append("sanctuaries: " + ", ".join(sanctuaries))
        summary = "; ".join(bits)

        return EnvContext(
            tool="sighting_context",
            ok=True,
            summary=summary,
            citations=[Citation(kind="sighting", source="OBIS-SEAMAP oceano (ETOPO1 / OISST)",
                                ref=str(individual_id))],
            individual_id=individual_id,
            n=r["n"],
            depth_median_m=depth_med,
            depth_min_m=r.get("depth_min"),
            depth_max_m=r.get("depth_max"),
            shelf_fraction=shelf,
            sst_median_c=sst_med,
            sst_min_c=r.get("sst_min"),
            sst_max_c=r.get("sst_max"),
            regions=regions,
            sanctuaries=sanctuaries,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tool = SightingContextTool()

    # Whale 479 — the Monterey humpback from the F5 chain.
    ctx = tool.query(479)
    print(f"\n[sighting_context 479] ok={ctx.ok}")
    print(f"  {ctx.summary}")
    if ctx.ok and ctx.shelf_fraction is not None:
        lever = ("shallow-shelf habitat → heavy traffic here IS strike-relevant"
                 if ctx.shelf_fraction >= 0.5 else
                 "mostly deep water → traffic less strike-relevant")
        print(f"  F5 risk-modulation read: {lever}")
