"""summarizer.py — LLM summarization of PROSE. Tools are never summarized.

After the stream split, summarization has one job: turn aged human/AI PROSE into
durable framed summary HumanMessage(s). Tool results are NOT its concern — they're
handled on the tool side (brief_tools → the manifest) and never folded into a summary.

    tool traceability  → the tool manifest (a leading HumanMessage of ToolBriefs)
    prose traceability → the summary restates the salient facts in the agent's words

Specific tool-derived facts survive because the AGENT'S OWN prose restated them
("I found the top match is #479") — that prose is what we summarize; the full tool
payload stays untouched in the log.

TWO PIECES:
  - summarize_messages(...) — module logic: slice, FILTER to prose, call the model,
    frame the result as summary HumanMessage(s). Returns a list (topical segmentation
    is a saved point; for now it's always one).
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

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from core.context.chunking import Boundary, _is_prose
from core.context.protocols import Summarizer, TokenCounter


def summarize_messages(
        messages: Sequence[BaseMessage],
        boundary: Boundary,
        summarizer: Summarizer,
        counter: TokenCounter,
) -> list[HumanMessage]:
    """Summarize the PROSE in `boundary` into framed summary HumanMessage(s).

    Args:
        messages:   full log; sliced by boundary.start_index..end_index.
        boundary:   a prose-sized contiguous slice (chunking.next_boundary).
        summarizer: injected model adapter (Summarizer protocol).
        counter:    used to stamp each summary's token count in additional_kwargs.

    Filters the slice to prose and summarizes only that — tool messages in the range
    are briefed separately (compaction) and shown in the manifest, not re-described
    here. Each summary is framed as a HumanMessage (a custom role isn't wire-valid;
    see view_builder). Returns a list for future topical segmentation; today it's one.
    Input-only: never reads from or writes to `messages`.
    """
    slice_ = messages[boundary.start_index: boundary.end_index + 1]
    prose = [m for m in slice_ if _is_prose(m)]
    if not prose:
        return []  # next_boundary guarantees prose; defensive no-op otherwise

    summary_texts: list[str] = summarizer.summarize(prose)

    # frame each summary string as a wire-valid HumanMessage
    def make_summary_message(summary: str) -> HumanMessage:
        return HumanMessage(
            id=uuid4().hex,
            content=f"[Summary of earlier conversation]:\n{summary}",
            additional_kwargs={
                "kind": "summary",
                "tokens": counter.count_text(summary),
                "created_at": datetime.now(timezone.utc).isoformat()
            },
        )

    return [make_summary_message(text) for text in summary_texts]


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
