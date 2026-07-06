"""app — the entry/shell layer: process entry points that wire the platform
(core/) to the domain (agents/). Home of the composition root (build_context),
the application surface (ConversationService), and the two callers that drive it:
ocean_runner (HTTP transport + TurnRunner adapter) and debug_driver (no-server driver)."""
