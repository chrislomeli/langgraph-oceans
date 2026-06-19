"""sighting_lookup.py — Layer 3 tool: "where and when has this individual been seen?"

The hinge of the F5 multi-hop. `photo_id` gives an `individual_id`; this tool returns
that whale's sighting history AND — the load-bearing bit — its **computed `range_bbox`**
(the envelope over its sighting points). That bbox is what `vessel_traffic` reads next.
The whole point: step N+1's argument is *computed from* step N's result — a dependency you
cannot script in advance (the agentic seam, see ocean-mammal-conservation-vision.md).

Dumb-tool contract (contracts.py): surfaces records + derived geometry; makes no control-
flow decision. On an individual with no sightings it returns `ok=False`, never raises.

Data note: the catalog spans SE Alaska (feeding), Hawaii (breeding), and California. Only
CA-ranged whales fall inside the West Coast AIS raster, so the F5 chain produces traffic for
those; an Alaska/Hawaii whale's `range_bbox` is honestly "outside AIS coverage" downstream.
"""

import logging
from datetime import date

from pydantic import BaseModel

from tools.contracts import BBox, Citation, Filters, ToolResult
from stores.postgres import get_pg_gateway

log = logging.getLogger(__name__)


class SightingRecord(BaseModel):
    """One encounter: when, where, and a public thumbnail (the multimodal citation)."""

    sighting_id: int
    obs_date: date | None = None
    lat: float
    lon: float
    species: str | None = None
    source: str | None = None  # provider (the contributing program/catalog)
    thumb_ref: str | None = None  # public thumb_url


class SightingHistory(ToolResult):
    """Chronological sightings + the COMPUTED range_bbox that feeds vessel_traffic."""

    individual_id: int
    name: str | None = None  # common_name / name from the individuals hub
    records: list[SightingRecord] = []  # chronological (oldest first)
    range_bbox: BBox | None = None  # envelope over the points — the multi-hop seam
    first_seen: date | None = None
    last_seen: date | None = None
    n_sightings: int = 0


# location is geography(Point,4326); cast to geometry for ST_X/ST_Y. Oldest first so a
# reader sees the history in order; NULL dates sort last.
_SIGHTINGS_SQL = """
    SELECT s.sighting_id,
           s.obs_date,
           ST_Y(s.location::geometry) AS lat,
           ST_X(s.location::geometry) AS lon,
           s.species,
           s.provider,
           s.thumb_url
    FROM sightings s
    WHERE s.individual_id = %s
      AND s.location IS NOT NULL
    ORDER BY s.obs_date NULLS LAST, s.sighting_id
"""

_INDIVIDUAL_SQL = """
    SELECT COALESCE(common_name, name) AS name, first_seen, last_seen, n_sightings
    FROM individuals
    WHERE individual_id = %s
"""


def _core_bbox(lats: list[float], lons: list[float], core_pct: float) -> BBox:
    """The CORE range — central `core_pct` of sightings, per axis.

    A migratory whale's lifetime min/max bbox is a continent-spanning rectangle (e.g. a
    Monterey humpback with a handful of Mexico breeding sightings → a box over open ocean
    it never swam, so downstream traffic is meaningless). Trimming the tails keeps the
    *habitat*, not the *journey*. core_pct=1.0 gives the full extent.
    """
    def lo_hi(vals: list[float]) -> tuple[float, float]:
        vs = sorted(vals)
        n = len(vs)
        tail = (1.0 - core_pct) / 2.0
        return vs[min(n - 1, int(tail * n))], vs[min(n - 1, int((1.0 - tail) * n))]

    lat_lo, lat_hi = lo_hi(lats)
    lon_lo, lon_hi = lo_hi(lons)
    return BBox(lat_min=lat_lo, lat_max=lat_hi, lon_min=lon_lo, lon_max=lon_hi)


class SightingLookupTool:
    """Return an individual's sighting history + derived range. Open the pool once."""

    def __init__(self):
        self.gw = get_pg_gateway()
        log.info("SightingLookupTool ready")

    def query(
        self, individual_id: int, filters: Filters | None = None, core_pct: float = 0.9  # noqa: ARG002
    ) -> SightingHistory:
        # `core_pct` defines range_bbox as the central fraction of sightings (trims
        # migratory outliers; 1.0 = full extent). NOTE: `filters` is accepted to honor the
        # one-contract rule; date_range/region scoping is deferred — the seam is here so
        # "sightings since DATE" / region-scoped range is free when the agent wants it.
        rows = self.gw.fetch_rows(_SIGHTINGS_SQL, (individual_id,))
        if not rows:
            return self._empty(individual_id, "No sightings for this individual")

        records = [
            SightingRecord(
                sighting_id=r["sighting_id"],
                obs_date=r["obs_date"],
                lat=r["lat"],
                lon=r["lon"],
                species=r.get("species"),
                source=r.get("provider"),
                thumb_ref=r.get("thumb_url"),
            )
            for r in rows
        ]
        lats = [rec.lat for rec in records]
        lons = [rec.lon for rec in records]
        range_bbox = _core_bbox(lats, lons, core_pct)
        # how many sightings fall outside the core range (the trimmed migratory tail)?
        n_outside = sum(
            1 for rec in records
            if not (range_bbox.lat_min <= rec.lat <= range_bbox.lat_max
                    and range_bbox.lon_min <= rec.lon <= range_bbox.lon_max)
        )

        meta = self.gw.fetch_rows(_INDIVIDUAL_SQL, (individual_id,))
        m = meta[0] if meta else {}
        dates = [rec.obs_date for rec in records if rec.obs_date]
        first_seen = m.get("first_seen") or (min(dates) if dates else None)
        last_seen = m.get("last_seen") or (max(dates) if dates else None)

        span = f"{first_seen}–{last_seen}" if first_seen and last_seen else "dates unknown"
        trim = f"; {n_outside} outlying sightings trimmed from range" if n_outside else ""
        summary = (
            f"{len(records)} sightings of {m.get('name') or f'individual {individual_id}'} "
            f"({span}); core range (central {core_pct:.0%}) lat "
            f"{range_bbox.lat_min:.2f}–{range_bbox.lat_max:.2f}, "
            f"lon {range_bbox.lon_min:.2f}–{range_bbox.lon_max:.2f}{trim}"
        )
        # One citation per sighting (capped) — provenance the agent + synthesis can cite.
        citations = [
            Citation(kind="sighting", source=rec.source or "OBIS-SEAMAP", locator=rec.thumb_ref,
                     ref=str(individual_id))
            for rec in records[:10]
        ]
        return SightingHistory(
            tool="sighting_lookup",
            ok=True,
            summary=summary,
            citations=citations,
            individual_id=individual_id,
            name=m.get("name"),
            records=records,
            range_bbox=range_bbox,
            first_seen=first_seen,
            last_seen=last_seen,
            n_sightings=len(records),
        )

    @staticmethod
    def _empty(individual_id: int, why: str) -> SightingHistory:
        return SightingHistory(
            tool="sighting_lookup", ok=False, summary=why, individual_id=individual_id,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from tools.vessel_traffic import VesselTrafficTool

    # Individual 479 — a Monterey Bay humpback (188 sightings): hotspot + sanctuary + lanes.
    sl = SightingLookupTool()
    vt = VesselTrafficTool()

    print("\n=== F5 chain: photo_id → sighting_lookup → vessel_traffic ===")
    hist = sl.query(479)
    print(f"\n[sighting_lookup] ok={hist.ok}")
    print(f"  {hist.summary}")

    if hist.ok and hist.range_bbox:
        # THE multi-hop: vessel_traffic's argument is COMPUTED from sighting_lookup's result.
        traffic = vt.query(hist.range_bbox)
        print(f"\n[vessel_traffic over that range] ok={traffic.ok}  lane_overlap={traffic.lane_overlap}")
        print(f"  {traffic.summary}")
        if traffic.ok:
            print(f"\n  → {hist.name or 'this whale'}'s range carries {traffic.total_transits:,} "
                  f"vessel transits/yr"
                  + (f", inside {', '.join(traffic.overlapping_zones)}" if traffic.overlapping_zones else "")
                  + " — the F5 ship-strike signal, end to end on real data.")
