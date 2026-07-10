"""Deterministic unit tests for the summarization plumbing (no LLM, no DB, no cost).

These cover "everything except the agent": the trigger (chunking), the packaging
(summarize_messages), the orchestration (ContextManager.prepare), and the view
assembly (build_view). The one non-deterministic part — does the LLM produce a
*good* summary — is deliberately NOT here; that's an occasional live smoke test.

The whole toolkit is three fakes, because the module injects its collaborators
(the Summarizer / TokenCounter protocols) — so we pass fakes at those seams
instead of patching:

  • FakeSummarizer — a SPY: returns a canned list[str] AND records what it was
    handed, so we can assert we filtered to prose before calling the model.
  • FakeCounter   — counts CHARACTERS, so budgets are exact and legible
    ("content='x'*10" == 10 tokens).
  • H/A/Atc/T     — a message factory with explicit ids, so we can assert bookmarks.
"""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.context import ContextManager, ContextPolicy
from core.context.chunking import Boundary, _is_prose, index_after_bookmark, next_boundary
from core.context.summarizer import _render_transcript, summarize_messages
from core.context.view_builder import build_view


# ── fakes & factory ─────────────────────────────────────────────────────────

class FakeSummarizer:
    """Satisfies the Summarizer protocol. Canned return + spy on the input."""

    def __init__(self, returns: list[str]) -> None:
        self.returns = returns
        self.seen: list | None = None  # what our code actually handed the model

    def summarize(self, messages: list) -> list[str]:
        self.seen = list(messages)
        return self.returns


class FakeCounter:
    """Counts characters, so a message of content 'x'*N is exactly N 'tokens'."""

    def count_messages(self, messages) -> int:
        return sum(len(str(m.content)) for m in messages)

    def count_text(self, text: str) -> int:
        return len(text)


def H(i: str, n: int = 10) -> HumanMessage:
    return HumanMessage(id=i, content="x" * n)


def A(i: str, n: int = 10) -> AIMessage:
    return AIMessage(id=i, content="x" * n)


def Atc(i: str, cid: str) -> AIMessage:  # AIMessage that REQUESTS a tool (not prose)
    return AIMessage(id=i, content="", tool_calls=[{"name": "f", "args": {}, "id": cid, "type": "tool_call"}])


def T(i: str, cid: str, n: int = 10) -> ToolMessage:  # a tool RESULT (not prose)
    return ToolMessage(id=i, tool_call_id=cid, content="r" * n)


def ids(messages) -> list[str]:
    return [m.id for m in messages]


class FakeState:  # what prepare/build_view read off state
    def __init__(self, messages, summary_messages=None, bookmark=None):
        self.messages = messages
        self.summary_messages = summary_messages or []
        self.last_processed_message_id = bookmark


# ── _is_prose ────────────────────────────────────────────────────────────────

def test_is_prose_classifies_conversational_vs_tool():
    assert _is_prose(H("h")) is True
    assert _is_prose(A("a")) is True          # plain assistant text
    assert _is_prose(Atc("a", "c")) is False  # assistant tool REQUEST
    assert _is_prose(T("t", "c")) is False     # tool RESULT


# ── index_after_bookmark ─────────────────────────────────────────────────────

def test_index_after_bookmark_none_starts_at_zero():
    assert index_after_bookmark([H("h1"), A("a1")], None) == 0


def test_index_after_bookmark_starts_after_the_id():
    msgs = [H("h1"), A("a1"), H("h2")]
    assert index_after_bookmark(msgs, "a1") == 2  # a1 is index 1 → start at 2


def test_index_after_bookmark_missing_id_fails_loud():
    with pytest.raises(ValueError):
        index_after_bookmark([H("h1")], "nope")


# ── next_boundary (the trigger) ──────────────────────────────────────────────

def test_next_boundary_fires_when_prose_crosses_budget():
    msgs = [H("h1"), A("a1"), H("h2"), A("a2"), H("hot")]  # 10 chars each
    b = next_boundary(msgs, None, FakeCounter(), budget=15, live_tail_size=1)
    assert b is not None
    assert b.message_ids == ["h1", "a1"]   # h1(10) + a1(10) = 20 ≥ 15
    assert b.end_message_id == "a1"


def test_next_boundary_none_when_under_budget():
    msgs = [H("h1"), H("hot")]
    assert next_boundary(msgs, None, FakeCounter(), budget=50, live_tail_size=1) is None


def test_next_boundary_hot_fence_protects_recent_even_if_large():
    # the only message is inside the HOT window → nothing eligible, even at 100 chars
    msgs = [H("big", n=100)]
    assert next_boundary(msgs, None, FakeCounter(), budget=10, live_tail_size=1) is None


def test_next_boundary_tools_ride_along_but_do_not_drive_the_budget():
    # the 100-char tool RESULT must count ~0; it takes the prose a2 to cross budget
    msgs = [H("h1"), Atc("a1", "c1"), T("t1", "c1", n=100), A("a2"), H("hot")]
    b = next_boundary(msgs, None, FakeCounter(), budget=15, live_tail_size=1)
    assert b is not None
    assert b.message_ids == ["h1", "a1", "t1", "a2"]  # tools ride along in the range
    assert b.end_message_id == "a2"                    # ...but PROSE is what tripped it


def test_next_boundary_single_huge_prose_message_is_its_own_boundary():
    msgs = [H("big", n=100), A("a1")]
    b = next_boundary(msgs, None, FakeCounter(), budget=50, live_tail_size=1)
    assert b is not None and b.message_ids == ["big"]


# ── summarize_messages (packaging) ───────────────────────────────────────────

def _boundary_over(messages) -> Boundary:
    return Boundary(
        start_index=0,
        end_index=len(messages) - 1,
        start_message_id=messages[0].id,
        end_message_id=messages[-1].id,
        message_ids=ids(messages),
    )


def test_summarize_messages_frames_one_summary_and_filters_prose():
    msgs = [H("h1"), Atc("a1", "c1"), T("t1", "c1"), A("a2")]
    fake = FakeSummarizer(returns=["SUMMARY"])

    out = summarize_messages(msgs, _boundary_over(msgs), fake, FakeCounter())

    assert len(out) == 1
    m = out[0]
    assert isinstance(m, HumanMessage)
    assert m.additional_kwargs["kind"] == "summary"
    assert m.content.startswith("[Summary of earlier conversation]")
    assert "SUMMARY" in m.content
    assert m.additional_kwargs["tokens"] == len("SUMMARY")

    # THE money assertion: the model was handed ONLY prose (tools filtered out).
    assert fake.seen is not None
    assert ids(fake.seen) == ["h1", "a2"]
    assert all(_is_prose(x) for x in fake.seen)


def test_summarize_messages_produces_one_message_per_returned_summary():
    msgs = [H("h1"), A("a1")]
    out = summarize_messages(msgs, _boundary_over(msgs), FakeSummarizer(["A", "B"]), FakeCounter())
    assert len(out) == 2
    assert all(m.additional_kwargs["kind"] == "summary" for m in out)


def test_summarize_messages_no_prose_returns_empty_and_skips_the_model():
    msgs = [Atc("a1", "c1"), T("t1", "c1")]  # nothing summarizable
    fake = FakeSummarizer(["should-not-be-used"])
    out = summarize_messages(msgs, _boundary_over(msgs), fake, FakeCounter())
    assert out == []
    assert fake.seen is None  # never called the model


# ── ContextManager.prepare (orchestration) ───────────────────────────────────

def _manager(budget=15, live_tail=1) -> ContextManager:
    return ContextManager(
        FakeCounter(),
        {"oceans_agent": ContextPolicy(chunk_token_budget=budget, live_tail_size=live_tail)},
    )


def test_prepare_over_budget_returns_chunk_and_advances_bookmark():
    state = FakeState([H("h1"), A("a1"), H("h2"), A("a2"), H("hot")])
    out = _manager().prepare(state, "oceans_agent", FakeSummarizer(["S"]))

    assert set(out) == {"summary_messages", "last_processed_message_id"}
    assert out["last_processed_message_id"] == "a1"      # boundary end (h1+a1 crossed 15)
    assert len(out["summary_messages"]) == 1


def test_prepare_under_budget_is_a_noop():
    state = FakeState([H("h1"), H("hot")])
    assert _manager().prepare(state, "oceans_agent", FakeSummarizer(["S"])) == {}


def test_prepare_unknown_policy_fails_loud():
    state = FakeState([H("h1"), A("a1")])
    with pytest.raises(KeyError):
        _manager().prepare(state, "does-not-exist", FakeSummarizer(["S"]))


# ── build_view (assembly) ────────────────────────────────────────────────────

def test_build_view_places_summaries_then_the_tail_after_the_bookmark():
    msgs = [H("h1"), A("a1"), H("h2"), A("a2")]
    summaries = [HumanMessage(id="s1", content="[Summary of earlier conversation]:\n…")]
    view = build_view(msgs, summaries, "a1", live_tail_size=6, compaction_size_chars=1000)
    assert ids(view) == ["s1", "h2", "a2"]  # summary + everything after a1


def test_build_view_with_no_summaries_is_the_raw_messages():
    msgs = [H("h1"), A("a1"), H("h2")]
    view = build_view(msgs, [], None, live_tail_size=6, compaction_size_chars=1000)
    assert ids(view) == ["h1", "a1", "h2"]


# ── _render_transcript (guards the key-typo class of bug) ─────────────────────

def test_render_transcript_emits_message_id_type_content():
    out = _render_transcript([HumanMessage(id="id1", content="hello")])
    assert json.loads(out) == [{"message_id": "id1", "type": "human", "content": "hello"}]
