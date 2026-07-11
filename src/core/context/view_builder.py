"""view_builder.py — Build the ephemeral message list for one model call.

Computed fresh every turn, never written back to state["messages"]. This is what
makes "never mutate; send a compressed view" hold.

THE SEAM. The agent node calls build_view(state), prepends its own SystemMessage,
and invokes the model on the result. The view is used for THIS call only and is
never written back to state["messages"].

THE MODEL — the view is [manifest] + [summaries] + [tail], chronological, assembled
fresh from the durable shadow state (message_summaries, tool_calls) + the raw log:

  • MANIFEST (leading standing block): the tool briefs rendered as ONE framed
    HumanMessage at the FRONT (see _render_manifest). It's an INDEX the agent scans
    ("have I called this?"), so it sits at a fixed, prominent spot — NOT folded into
    the system prompt, which must stay byte-stable for prompt caching.
  • SUMMARIES (inline): the COLD region (before the bookmark) as framed
    HumanMessages in place, at normal salience — a summary is compressed HISTORY,
    not an instruction, so it reads as history where its turns sat, not hoisted to
    system. History → inline; index → standing.
  • TAIL: everything after the bookmark, kept raw.

WIRE CAVEAT: a framed block needs a valid role at send time — a custom
ChatMessage(role="summary"/"manifest") is NOT wire-valid. Use a HumanMessage so it
ships as a normal turn. (A mid-conversation SystemMessage keeps position but
reintroduces system priority and muddies caching — prefer the framed turn.)

THE HARD CEILING. When view_token_ceiling + counter are supplied, the three layers
are fit to budget first (_fit_to_ceiling): the tail is protected, then summaries,
then briefs shed — EPHEMERALLY (the outgoing copy only; state is never touched, so
dropped items return next turn). Pair-safety is local to _fit_tail (drop a leading
orphan ToolMessage). With no ceiling, every layer is kept.

NEVER MUTATE. build_view is a pure function of state → list; the ground-truth log
grows untouched. With no summaries, no briefs, and a short tail, the view is just
the raw messages.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from core.context.chunking import index_after_bookmark
from core.context.protocols import TokenCounter
from core.context.state import ToolBrief


# ── the view-fitter (the HARD ceiling; ephemeral, never touches state) ────────
#
# The fitter is layer-AWARE by necessity (see the design discussion): it can't
# reason over a flat list because (a) the cascade priority — shed manifest, then
# summaries, then tail — needs to know which items are which, and (b) each layer
# shrinks by its OWN rule and has its OWN representation (manifest is a dict → one
# rendered block; summaries are framed HumanMessages; the tail is raw messages with
# pair-safety + a protected current turn). So it takes the THREE components, not the
# assembled view, and returns the fitted three components for build_view to flatten.



def _fit_to_ceiling(
    summaries: list[HumanMessage],
    tail: list[BaseMessage],
    tool_call_values: list[ToolBrief],
    ceiling: int,
    counter: TokenCounter,
) -> tuple[list[HumanMessage], list[BaseMessage], list[str]]:
    """Fit the three view layers under `ceiling` by FILLING toward the budget in
    strict priority order — tail, then summaries, then tool briefs.

    Priority == survival order. Each layer keeps what fits in the budget LEFT OVER by
    the layers above it, so the tail (warmest, incl. the current turn) is filled first
    and shrinks last; tool briefs (coldest) get only the remainder and shrink first.
    That's the same outcome as the "shed manifest → summaries → tail" cascade, just
    expressed as allocation instead of subtraction — and still minimum-shed, since each
    layer keeps the NEWEST run that fits and drops only the oldest overflow.

    Pure / ephemeral: shrinks the outgoing copies only, never state — dropped items are
    still in state and reappear next turn.

    Returns the fitted (summaries, tail, briefs). Propagates ContextOverflowError from
    `_fit_tail` when the current turn alone won't fit — the one unrecoverable case.
    """
    remaining = ceiling

    # Priority 1: the last turn is non-negotiable — if it alone busts the ceiling, that's fatal.
    kept_tail, remaining = _fit_tail(tail, remaining, counter)

    # Priority 2: summaries, newest-first retention.
    kept_summaries, remaining = _fit_summaries(summaries, remaining, counter)

    # Priority 3: tool briefs get whatever is left.
    kept_briefs, remaining = _fit_manifest(tool_call_values, remaining, counter)

    return kept_summaries, kept_tail, kept_briefs



def _fit_manifest(
    tool_call_values: list[ToolBrief],
    budget: int,
    counter: TokenCounter,
) -> tuple[list[str], int]:
    """Keep the newest contiguous run of briefs that fits in `budget`.

    Walk newest → oldest; stop at the first brief that doesn't fit, so the kept
    set is always a contiguous suffix of the call history (no gaps).
    Returns (rendered briefs in chronological order, budget left over).
    """
    kept: list[tuple[int, str]] = []
    for brief in sorted(tool_call_values, key=lambda b: b.ordinal, reverse=True):
        text = brief.model_dump_json()
        sz = counter.count_text(text)
        if sz > budget:
            break
        kept.append((brief.ordinal, text))
        budget -= sz
    kept.sort()  # back to chronological order for rendering
    return [text for _, text in kept], budget



def _fit_summaries(
    summaries: list[HumanMessage],
    budget: int,
    counter: TokenCounter,
) -> tuple[list[HumanMessage], int]:
    """Layer 2. Keep the NEWEST summaries that fit in `budget`; drop the oldest overflow.

    Walk newest → oldest, keeping until the next summary won't fit, then restore
    chronological order. Returns (kept summaries, budget left over).

    EPHEMERAL: dropped summaries stay in state for next turn — durable
    summary-of-summaries (recompression) is the MEMORY axis, not here.
    """
    kept: list[HumanMessage] = []
    for message in reversed(summaries):
        sz = counter.count_messages([message])
        if sz > budget:
            break
        kept.append(message)
        budget -= sz
    kept.reverse()
    return kept, budget


def _fit_tail(
    messages: list[BaseMessage],
    budget: int,
    counter: TokenCounter,
) -> tuple[list[BaseMessage], int]:
    """Layer 1 (warmest). Keep the current turn (mandatory) + as many older messages fit.

    The current turn = from the last HumanMessage to the end; it is NON-NEGOTIABLE. If it
    alone exceeds `budget`, raise ContextOverflowError — there's nothing left to drop and
    answering without it is pointless (case-4 last resort; truncation would live here if
    we chose to add it).

    Older messages are kept newest-first and contiguous (a suffix), so no interior gaps.
    Pair-safety: after trimming, drop any leading ToolMessage(s) whose
    AIMessage(tool_calls) fell outside the budget — a view can't START on an orphan tool
    result. (A kept AIMessage(tool_calls) always retains its ToolMessage: the result is
    newer, so it was considered first and kept.)

    Returns (older-kept + current turn, budget left over).
    """
    # find the last turn and keep at least that as mandatory
    turn_start = next(
        (i for i in reversed(range(len(messages))) if isinstance(messages[i], HumanMessage)),
        0,
    )
    mandatory = messages[turn_start:]
    mandatory_size = counter.count_messages(mandatory)
    if mandatory_size > budget:
        raise ContextOverflowError(
            f"current turn ({mandatory_size} tokens) exceeds tail budget ({budget})"
        )
    budget -= mandatory_size

    # Older messages: newest-first, contiguous, pair-safe — trim freely.
    kept: list[BaseMessage] = []
    for message in reversed(messages[:turn_start]):
        sz = counter.count_messages([message])
        if sz > budget:
            break
        kept.append(message)
        budget -= sz
    kept.reverse()
    while kept and isinstance(kept[0], ToolMessage):
        budget += counter.count_messages([kept[0]])
        kept.pop(0)

    return kept + mandatory, budget


def _render_manifest(briefs: list[str]) -> HumanMessage | None:
    """Frame the surviving tool briefs as ONE leading, standing reference message.

    Each brief is already a JSON object string (ToolBrief.model_dump_json). Returns a
    single framed HumanMessage — placed FIRST in the view so it's the most prominent,
    consistently-located block the agent scans for "have I already called this?" — or
    None when there are no briefs.

    NOT folded into the system prompt: the manifest changes every turn, and the system
    prompt must stay byte-stable so Anthropic can cache it. A leading framed turn keeps
    the manifest out of the cached prefix while still sitting at a fixed, findable spot.
    Rebuilt every turn; never persisted.
    """
    if not briefs:
        return None
    body = "\n".join(briefs)
    return HumanMessage(content=f"[Tools already called this session]\n{body}")


def build_view(
    messages: Sequence[BaseMessage],
    message_summaries: Sequence[HumanMessage],
    tool_calls: dict[str, ToolBrief],
    last_processed_message_id: str | None,
    counter: TokenCounter | None = None,
    view_token_ceiling: int | None = None,
) -> list[BaseMessage]:
    """Assemble the outgoing view: [manifest] + [summaries] + [tail]. Pure; never mutates state.

    Args:
        messages:                   full log (input only; never mutated).
        message_summaries:          ordered COLD prose summaries (framed HumanMessages).
        tool_calls:                 the tool manifest (id → brief); rendered as the
                                    leading standing block (see _render_manifest).
        last_processed_message_id:  prose is summarized up to here; the tail is
                                    everything after it.
        counter:                    token estimator, injected. Only needed to FIT.
        view_token_ceiling:         the HARD ceiling. None ⇒ fitting OFF (assemble as-is,
                                    every brief/summary/tail kept). This is the per-agent
                                    policy's dial — a STARTING POINT the fitter enforces
                                    as a backstop when a turn overflows it.

    The manifest LEADS the view (standing dedup index), then inline summaries, then the
    tail. When `view_token_ceiling` and `counter` are both set, the three layers are fit
    to the budget first (see `_fit_to_ceiling`); otherwise everything is kept.
    """

    start = index_after_bookmark(messages, last_processed_message_id)
    tail = list(messages[start:])
    summaries = list(message_summaries)
    tool_call_values = list(tool_calls.values())

    # Default (fitting OFF): every brief is rendered, in call order.
    briefs = [b.model_dump_json() for b in tool_call_values]

    # HARD ceiling (ephemeral): fit the layers to the budget. Dormant until the policy
    # sets view_token_ceiling AND a counter is injected — so the normal (in-budget) path
    # keeps every brief/summary/tail message, unchanged.
    if view_token_ceiling is not None and counter is not None:
        summaries, tail, briefs = _fit_to_ceiling(
            summaries=summaries,
            tail=tail,
            tool_call_values=tool_call_values,
            ceiling=view_token_ceiling,
            counter=counter,
        )

    # The manifest is a STANDING index → one framed HumanMessage at the FRONT, rebuilt
    # every turn at a fixed spot (NOT in the system prompt — that stays cache-stable).
    manifest = _render_manifest(briefs)
    lead = [manifest] if manifest is not None else []
    return [*lead, *summaries, *tail]
