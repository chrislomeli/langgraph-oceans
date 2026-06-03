"""
world-simiulator.prompts

File-based, versioned prompt registry built on Jinja2.

Public API
──────────
    from prompts import PromptRegistry

    registry = PromptRegistry()
    registry.register_model(MyModel)
    text = registry.render("evaluate", {"cluster_id": "north", ...})

The registry is constructed by the composition root (``main.py`` / app
bootstrap) and passed to graph builders. Importing it here from
``prompts.registry`` keeps the public import path stable while the
implementation lives in its own module.

The dependency arrow inside this project is always ``agent → prompt``.
This package never imports from ``agents/`` — agents register the
Pydantic models they want their templates to reference at startup.
"""

from prompts.registry import PromptRegistry

__all__ = ["PromptRegistry"]
