"""agents/tools.py — B-BIND: make the Layer-3 tools LLM-callable.

The agents (B5) can't call the `*.query()` classes directly — an LLM picks a tool by
reading its **ad** (the name + description + arg schema). This module wraps each built
tool as a LangChain `@tool` so it can be bound to the model.

═══════════════════════════════════════════════════════════════════════════════════
TIER SPLIT (see memory: agentic-division-of-labor)
  • Claude scaffolded the plumbing below — lazy tool singletons, the `@tool` wrappers,
    arg shapes, the F5-chain seam (sighting_lookup emits the bbox; vessel_traffic takes
    4 floats so the LLM can thread one into the next).
  • TODO(you): write the AD for each tool — the docstring. The docstring IS what the
    LLM reads to decide when to call it. The bodies are done; the ads are the learning.

THE AD TEMPLATE (from docs/design/agents-orchestration-design.md). Each docstring =
    <Trigger>   — when SHOULD the agents reach for this tool? (the user-intent it serves)
    <Anti-overlap> — when should it NOT (vs the neighbouring tool it's confusable with)?
    <Returns>   — what comes back, in terms the agents can act on / chain.
Keep it tight and concrete; the LLM reads these literally. The `Args:` lines feed the
arg schema, so describe each parameter too.
═══════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from langchain_core.tools import tool

# ── lazy singletons: building a tool opens the pool (and photo_id loads the image
#    embedder), so construct on first use, not at import. ──────────────────────────
_photo = _sight = _ctx = _vessel = None


def _photo_tool():
    global _photo
    if _photo is None:
        from tools.photo_id import PhotoIDTool
        _photo = PhotoIDTool()
    return _photo


def _sighting_tool():
    global _sight
    if _sight is None:
        from tools.sighting_lookup import SightingLookupTool
        _sight = SightingLookupTool()
    return _sight


def _context_tool():
    global _ctx
    if _ctx is None:
        from tools.sighting_context import SightingContextTool
        _ctx = SightingContextTool()
    return _ctx


def _vessel_tool():
    global _vessel
    if _vessel is None:
        from tools.vessel_traffic import VesselTrafficTool
        _vessel = VesselTrafficTool()
    return _vessel


# ── the four LLM-callable tools (bodies done; docstrings = TODO(you) ad copy) ──────

@tool
def photo_id(image_path: str) -> str:
    """Identify which catalogued whale an individual fluke photo shows.

    Trigger: a whale photo is provided and you need to know WHO it is — the entry
        point for almost any question about a specific pictured whale.
    Anti-overlap: requires a photo; without one, do not call this. It answers only
        "which individual" — NOT where/when it's been seen (-> sighting_lookup) or its
        habitat conditions (-> sighting_context). Call those after, with the id here.
    Returns: ranked candidate individual(s) + confidence + `margin` (how clearly #1
        beats #2) + an abstain/NOVEL flag when nothing is confidently close. Treat
        NOVEL as a SOFT prior, not a verdict — corroborate with the margin and
        sighting/location evidence before concluding "new whale." Yields the
        individual_id the sighting tools need next.

    Args:
        image_path: local path or URL to the query whale-fluke photo.
    """
    r = _photo_tool().query_by_image(image_path)
    return r.summary


@tool
def sighting_lookup(individual_id: int) -> str:
    """Look up a known individual's sighting history and geographic range.

    Trigger: you have an individual_id (from photo_id) and need WHERE/WHEN it has been
        seen, or its range in order to check ship traffic over its waters.
    Anti-overlap: this is the WHERE/WHEN + RANGE tool. For the *physical conditions* of
        its habitat (depth, temperature, sanctuary) use sighting_context instead. Both
        take individual_id, so choose by intent: locations + range here, habitat
        character there.
    Returns: chronological sightings, first/last seen, count, and the core range as a
        bounding box (lat/lon min/max). Pass those four numbers straight to
        vessel_traffic to measure ship-strike exposure.

    Args:
        individual_id: the catalogued whale id (from photo_id).
    """
    r = _sighting_tool().query(individual_id)
    if not r.ok or r.range_bbox is None:
        return r.summary
    b = r.range_bbox
    # surface the bbox as plain numbers so the agents can pass them to vessel_traffic
    # (the F5 multi-hop seam — output of step N becomes the argument of step N+1).
    return (f"{r.summary}\n"
            f"range_bbox: lat_min={b.lat_min:.4f} lat_max={b.lat_max:.4f} "
            f"lon_min={b.lon_min:.4f} lon_max={b.lon_max:.4f}")


@tool
def sighting_context(individual_id: int) -> str:
    """Summarize the environmental conditions of a known individual's habitat.

    Trigger: you have an individual_id and want to characterize WHERE IT LIVES — water
        depth, temperature, whether it's over the continental shelf, which
        sanctuaries/regions — e.g. to judge how strike- or habitat-sensitive its
        waters are.
    Anti-overlap: this is the CONDITIONS tool, not the locations tool. For the list of
        sightings or the range box that feeds vessel_traffic, use sighting_lookup. Both
        take individual_id; choose by intent: habitat character here, where/when +
        range there.
    Returns: median/min/max depth, shelf fraction (share of sightings over the <200 m
        shelf — a ship-strike modulation signal), SST range, and the distinct
        regions/sanctuaries the whale frequents.

    Args:
        individual_id: the catalogued whale id (from photo_id).
    """
    return _context_tool().query(individual_id).summary


@tool
def vessel_traffic(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                   year: int = 2024) -> str:
    """Measure ship traffic over a geographic area and whether it's a managed speed zone.

    Trigger: you have a bounding box (from sighting_lookup's range) and need ship-strike
        EXPOSURE — how heavy vessel traffic is there and whether protections apply.
    Anti-overlap: takes a geographic box, NOT an individual_id — get the box from
        sighting_lookup first. Reports raw traffic numbers only; YOU judge whether that
        constitutes "risk" (the tool does not decide).
    Returns: total transits, mean transits per trafficked cell, busiest cell, and
        lane_overlap = whether the area intersects an NMS/VSR speed-reduction zone (with
        the zone names). Heavy traffic + a shallow-shelf habitat (see sighting_context)
        + lane overlap = the ship-strike picture.

    Args:
        lat_min: south edge of the range box.
        lat_max: north edge of the range box.
        lon_min: west edge of the range box.
        lon_max: east edge of the range box.
        year: AIS year to read (2022–2025; default 2024).
    """
    from tools.contracts import BBox
    bbox = BBox(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
    return _vessel_tool().query(bbox, year=year).summary


# The roster the B5 graph will bind to the model. (stock_facts / doc_search join here
# once built; obis_density optional.)
TOOLS = [photo_id, sighting_lookup, sighting_context, vessel_traffic]


if __name__ == "__main__":
    # Smoke: confirm the four are LLM-callable (name + schema the model will see).
    # Run: uv run python -m agents.tools
    for t in TOOLS:
        print(f"\n=== {t.name} ===")
        print(f"  description: {t.description!r}")
        print(f"  args: {list(t.args.keys())}")
