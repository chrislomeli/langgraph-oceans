"""core.context — Hand-rolled context management for the ReAct loop.

Assemble-before-every-call context compression that never mutates the message
log. See docs/design/Summarization.md for the full spec.

Public surface. Everything else (chunking internals, compaction policy, the
summarizer adapter) is reachable but not re-exported here — import from the
submodule when you need it.
"""

from core.context.manager import ContextPolicy, ContextManager
from core.context.protocols import Summarizer, TokenCounter, ToolResultStore
from core.context.retention import RetentionPolicy
from core.context.state import ContextStateFields, ToolBrief
from core.context.summarizer import LLMSummarizer
from core.context.token_counter import HeuristicTokenCounter


__all__ = [
    "ContextManager",
    "ContextPolicy",
    "ContextStateFields",
    "HeuristicTokenCounter",
    "LLMSummarizer",
    "RetentionPolicy",
    "ToolBrief",
    "TokenCounter",
    "Summarizer",
    "ToolResultStore",
]
