"""tools — Layer 3: the agent's typed, deliberately "dumb" capabilities.

Each tool speaks the shared contract in `contracts.py` (one `Filters` shape in, a
`ToolResult` subclass out) and never makes a control-flow decision — it surfaces
signals and lets the agent (Layer 2) decide. `photo_id` is the perception tool;
`hybrid_search` / `sighting_lookup` / `vessel_traffic` / `catalog_search` follow.
"""
