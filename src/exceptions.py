"""
world-simiulator.exceptions

Project-wide exception hierarchy.

Why a hierarchy?
────────────────
Code that catches errors should be able to distinguish a transport failure
from an agent failure from a config failure without parsing message strings.
Bare ``ValueError`` and ``KeyError`` are too coarse for that.

All project-raised exceptions inherit from :class:`OgarError` so callers
that want a catch-all can do::

    try:
        ...
    except OgarError as exc:
        handle(exc)

while callers that care about a specific category can narrow the catch::

    try:
        ...
    except TransportError as exc:
        retry_or_dlq(exc)

When to use which
─────────────────
TransportError — wire-format problems, queue/topic failures, schema mismatches
                 detected at the transport boundary.
AgentError     — failures inside a graph: bad state, routing dead ends,
                 reducer invariant violations, LLM output that cannot be parsed.
ResourceError  — invalid resource state transitions or capacity violations.
ConfigError    — missing or malformed settings, unknown LLM role/label,
                 misconfigured registry entries.
PromptError    — unknown prompt name, missing required template variables,
                 unknown model in a schema filter.
"""

from __future__ import annotations


class OgarError(Exception):
    """Base class for all project-raised exceptions."""


class TransportError(OgarError):
    """Wire-level or transport-layer failure (queue, topic, envelope)."""


class AgentError(OgarError):
    """Failure inside an agent graph (state, routing, classification)."""


class ResourceError(OgarError):
    """Invalid resource transition or capacity violation."""


class ConfigError(OgarError):
    """Settings, registry, or model-catalog misconfiguration."""


class PromptError(OgarError):
    """Prompt registry or template rendering failure."""
