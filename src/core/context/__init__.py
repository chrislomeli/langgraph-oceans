"""core.context — Hand-rolled context management for the ReAct loop.

Assemble-before-every-call context compression that never mutates the message
log. See docs/design/Summarization.md for the full spec.

Public surface. Everything else (chunking internals, compaction policy, the
summarizer adapter) is reachable but not re-exported here — import from the
submodule when you need it.
"""

from core.context.manager import ContextPolicy, ContextManager
from core.context.manifest import build_tool_manifest
from core.context.protocols import Summarizer, TokenCounter, ToolResultStore
from core.context.retention import RetentionPolicy


__all__ = [
    "ContextManager",
    "ContextPolicy",
    "RetentionPolicy",
    "build_tool_manifest",
    "TokenCounter",
    "Summarizer",
    "ToolResultStore",
]
