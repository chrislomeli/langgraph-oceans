"""summarizer.py — LLM summarization of PROSE. Tools are never summarized.

After the stream split, summarization has one job: turn aged human/AI PROSE into
durable SummaryChunk(s). Tool results are NOT its concern — they're shown as
manifest lines (extract + ref) and never folded into a summary. So there's no
ref-preservation contract here: summaries carry no refs.

    tool traceability  → manifest (build_tool_manifest) + resolve_full_tool_result
    prose traceability → SummaryChunk.message_ids point straight at the source

Specific tool-derived facts survive because the AGENT'S OWN prose restated them
("I found the top match is #479") — that prose is what we summarize; the full tool
payload stays one manifest ref away.

TWO PIECES:
  - summarize_chunk(...)  — module logic: slice, FILTER to prose, call the model,
    package into SummaryChunk(s). Returns a list (segmentation is a saved point;
    for now it's always one chunk).
  - LLMSummarizer         — the injected model adapter. It holds the model AND the
    "summarizer" system prompt (rendered from the prompt registry in the graph and
    passed in — the structured-field instructions live in THAT template, not here,
    so core stays free of domain prompt text).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from core.context.protocols import Summarizer, TokenCounter
from core.context.chunking import Boundary, _is_prose
from core.context.state import SummaryChunk


def summarize_messages(
        messages: Sequence[BaseMessage],
        boundary: Boundary,
        summarizer: Summarizer,
        counter: TokenCounter,
) -> list[HumanMessage]:
    """Summarize the PROSE in `boundary` into one SummaryChunk (list for future
    segmentation).

    Args:
        messages:   full log; sliced by boundary.start_index..end_index.
        boundary:   a prose-sized contiguous slice (chunking.next_boundary).
        summarizer: injected model adapter (Summarizer protocol).
        counter:    fills token_count_estimate = tokens of the produced summary_text.

    The chunk's message_ids are the WHOLE contiguous range (matches start/end and
    the advancing bookmark); the summary TEXT is of the prose within — tool messages
    in the range are covered by the manifest, not re-described here. Input-only:
    never reads from or writes to `messages`.
    """
    slice_ = messages[boundary.start_index: boundary.end_index + 1]
    prose = [m for m in slice_ if _is_prose(m)]
    if not prose:
        return []  # next_boundary guarantees prose; defensive no-op otherwise

    summary_chunks: list[str] =  summarizer.summarize(prose)

    # transform to human messages
    def make_summary_message(summary: str) -> HumanMessage:
        return HumanMessage(
            id= uuid4().hex,
            content=f"[Summary of earlier conversation]:\n{summary}",
            additional_kwargs={
                "kind": "summary",
                "tokens": counter.count_text(summary),
                "created_at": datetime.now(timezone.utc).isoformat()
            },
        )

    summary_messages = [make_summary_message(c) for c in summary_chunks]
    return summary_messages


class LLMSummarizer:
    """Summarizer adapter over an injected chat model + its system prompt.

    Constructed in the graph with a cheap/fast model from the LLM registry and the
    "summarizer" system prompt (which carries the structured-field instructions).
    Structurally satisfies the Summarizer protocol.
    """

    def __init__(self, model, system_prompt: str) -> None:
        self._model = model
        self._system_prompt = system_prompt

    def summarize(self, messages: list[BaseMessage]) -> list[str]:
        prompt = [
            SystemMessage(self._system_prompt),
            HumanMessage(f"Conversation to summarize:\n\n{_render_transcript(messages)}"),
        ]

        result = self._model.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        # One summary for now; N-way topical segmentation is a saved point (splitting
        # one model response into many needs a delimiter/structured-output protocol).
        return [content if isinstance(content, str) else str(content)]


def _render_transcript(messages: Sequence[BaseMessage]) -> str:
    """Flatten prose messages into a labeled transcript for the summarizer prompt.

    Rendered to text (rather than replayed as chat turns) so the model summarizes
    the conversation instead of trying to continue it.
    """
    prose = [dict(message_id=m.id, type=m.type, content=m.content) for m in messages]
    return json.dumps(prose)
