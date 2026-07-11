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
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.context import ContextManager, ContextPolicy
from core.context.chunking import Boundary, _is_prose, index_after_bookmark, next_boundary
from core.context.compaction import brief_tools
from core.context.state import ContextStateFields, ToolBrief, update_dict
from core.context.summarizer import LLMSummarizer, _render_transcript, summarize_messages
from core.context.view_builder import (
    _fit_manifest,
    _fit_summaries,
    _fit_tail,
    _fit_to_ceiling,
    build_view,
)


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


def Atc(i: str, cid: str, n: int = 0) -> AIMessage:  # AIMessage that REQUESTS a tool (not prose)
    # n>0 gives it prose content too, so it can be dropped-by-size in fitter tests.
    return AIMessage(id=i, content="x" * n, tool_calls=[{"name": "f", "args": {}, "id": cid, "type": "tool_call"}])


def T(i: str, cid: str, n: int = 10) -> ToolMessage:  # a tool RESULT (not prose)
    return ToolMessage(id=i, tool_call_id=cid, content="r" * n)


def ids(messages) -> list[str]:
    return [m.id for m in messages]


class FakeModel:  # stands in for a chat model behind LLMSummarizer
    def __init__(self, content: str) -> None:
        self.content = content
        self.seen = None  # the prompt it was invoked with

    def invoke(self, prompt):
        self.seen = prompt
        return AIMessage(content=self.content)


class FakeState:  # what prepare/build_view read off state (attr names match OceanState)
    def __init__(self, messages, message_summaries=None, bookmark=None, tool_calls=None):
        self.messages = messages
        self.message_summaries = message_summaries or []
        self.last_processed_message_id = bookmark
        self.tool_calls = tool_calls or {}  # prepare seeds brief ordinals from len(this)


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


def test_next_boundary_starts_after_the_bookmark():
    msgs = [H("h1"), A("a1"), H("h2"), A("a2"), H("hot")]
    b = next_boundary(msgs, "a1", FakeCounter(), budget=15, live_tail_size=1)
    assert b is not None
    assert b.message_ids == ["h2", "a2"]   # only the region AFTER a1
    assert b.end_message_id == "a2"


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

    assert set(out) == {"tool_calls", "message_summaries", "last_processed_message_id"}
    assert out["last_processed_message_id"] == "a1"      # boundary end (h1+a1 crossed 15)
    assert len(out["message_summaries"]) == 1
    assert out["tool_calls"] == {}                        # prose-only slice → no tool briefs


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
    # build_view(messages, message_summaries, tool_calls, last_processed_message_id)
    view = build_view(msgs, summaries, {}, "a1")
    assert ids(view) == ["s1", "h2", "a2"]  # summary + everything after a1


def test_build_view_with_no_summaries_is_the_raw_messages():
    msgs = [H("h1"), A("a1"), H("h2")]
    view = build_view(msgs, [], {}, None)
    assert ids(view) == ["h1", "a1", "h2"]


def test_build_view_leads_with_the_tool_manifest():
    msgs = [H("h1"), A("a1"), H("h2")]
    tool_calls = {
        "c1": ToolBrief(id="c1", ordinal=0, name="photo_id", query={"image": "H8"}, summary="3 cands"),
    }
    view = build_view(msgs, [], tool_calls, None)

    # the manifest LEADS the view as one framed HumanMessage...
    lead = view[0]
    assert isinstance(lead, HumanMessage)
    assert "Tools already called" in lead.content
    assert "photo_id" in lead.content        # the brief is rendered into it
    # ...followed by the raw tail (lead has an auto-id, so slice it off).
    assert ids(view[1:]) == ["h1", "a1", "h2"]


# ── _render_transcript (guards the key-typo class of bug) ─────────────────────

def test_render_transcript_emits_message_id_type_content():
    out = _render_transcript([HumanMessage(id="id1", content="hello")])
    assert json.loads(out) == [{"message_id": "id1", "type": "human", "content": "hello"}]


# ── brief_tools (tool-side compaction) ───────────────────────────────────────

def test_brief_tools_briefs_and_merges_a_full_pair():
    msgs = [Atc("a1", "c1"), T("t1", "c1", n=5)]
    briefs = brief_tools(msgs, _boundary_over(msgs), max_chars=10, base_ordinal=0)

    assert set(briefs) == {"c1"}
    b = briefs["c1"]
    assert b.name == "f"          # from the AIMessage half
    assert b.query == {}          # from the AIMessage half
    assert b.summary == "rrrrr"   # from the ToolMessage half (5 chars ≤ 10) — merged, not overwritten
    assert b.ordinal == 0


def test_brief_tools_truncates_summary_and_tolerates_a_partial_pair():
    msgs = [T("t1", "c1", n=100)]  # tool result with no AIMessage half in the slice
    b = brief_tools(msgs, _boundary_over(msgs), max_chars=10)["c1"]
    assert len(b.summary) == 10   # truncated to max_chars
    assert b.name is None         # the missing half is left None


def test_brief_tools_ordinals_are_dense_and_seeded_across_sweeps():
    # THE ordinal-bug regression: two calls, ordinals must be dense +1 and seeded from
    # base_ordinal — NOT the slice index (which would be 0 and 3 here and reset per sweep).
    msgs = [Atc("a1", "c1"), T("t1", "c1"), A("mid"), Atc("a2", "c2"), T("t2", "c2")]
    briefs = brief_tools(msgs, _boundary_over(msgs), max_chars=10, base_ordinal=5)
    assert briefs["c1"].ordinal == 5
    assert briefs["c2"].ordinal == 6


def test_brief_tools_empty_when_slice_has_no_tools():
    msgs = [H("h1"), A("a1")]
    assert brief_tools(msgs, _boundary_over(msgs), max_chars=10) == {}


# ── the fitter: _fit_tail (warmest layer + case-4 last resort) ────────────────

def test_fit_tail_keeps_current_turn_plus_older_that_fit():
    msgs = [H("h1"), A("a1"), H("hcur")]   # 10 chars each
    kept, budget = _fit_tail(msgs, budget=25, counter=FakeCounter())
    # hcur (current turn, mandatory)=10 → 15 left; a1(10) fits → 5 left; h1(10) doesn't.
    assert ids(kept) == ["a1", "hcur"]
    assert budget == 5


def test_fit_tail_raises_when_current_turn_alone_exceeds_budget():
    msgs = [H("hcur", n=100)]
    with pytest.raises(ContextOverflowError):
        _fit_tail(msgs, budget=50, counter=FakeCounter())


def test_fit_tail_drops_a_leading_orphan_tool_message():
    # Atc has content (n=10) so it can be dropped by size; its ToolMessage would then
    # lead the tail as an orphan → must be dropped for pair-safety.
    msgs = [Atc("a0", "c1", n=10), T("t1", "c1", n=5), H("hcur", n=10)]
    kept, _ = _fit_tail(msgs, budget=16, counter=FakeCounter())
    # 16 - hcur(10) = 6; T(5) fits → 1 left; Atc(10) doesn't → orphan T popped.
    assert ids(kept) == ["hcur"]


# ── the fitter: _fit_summaries ───────────────────────────────────────────────

def test_fit_summaries_keeps_newest_that_fit_drops_oldest():
    summaries = [H("s1"), H("s2"), H("s3")]  # 10 each
    kept, budget = _fit_summaries(summaries, budget=25, counter=FakeCounter())
    assert ids(kept) == ["s2", "s3"]  # oldest (s1) dropped; chronological order restored
    assert budget == 5


# ── the fitter: _fit_manifest ────────────────────────────────────────────────

def _brief(cid: str, ordinal: int) -> ToolBrief:
    return ToolBrief(id=cid, ordinal=ordinal, name="f", query={}, summary="s")


def test_fit_manifest_empty_budget_keeps_nothing():
    kept, budget = _fit_manifest([_brief("c1", 0)], budget=0, counter=FakeCounter())
    assert kept == [] and budget == 0


def test_fit_manifest_keeps_newest_by_ordinal_in_chronological_order():
    briefs = [_brief("c0", 0), _brief("c1", 1), _brief("c2", 2)]  # equal-sized JSON
    size = FakeCounter().count_text(briefs[0].model_dump_json())
    kept, _ = _fit_manifest(briefs, budget=2 * size, counter=FakeCounter())
    # only two fit → the two NEWEST (ordinals 1,2), rendered oldest-first.
    assert [json.loads(t)["ordinal"] for t in kept] == [1, 2]


# ── the fitter: _fit_to_ceiling (priority: tail > summaries > briefs) ─────────

def test_fit_to_ceiling_under_budget_keeps_all_layers():
    summaries = [H("s1")]
    tail = [H("h1")]
    briefs = [_brief("c1", 0)]
    kept_s, kept_t, kept_b = _fit_to_ceiling(summaries, tail, briefs, ceiling=1000, counter=FakeCounter())
    assert ids(kept_s) == ["s1"]
    assert ids(kept_t) == ["h1"]
    assert len(kept_b) == 1


def test_fit_to_ceiling_sheds_briefs_before_summaries_before_tail():
    summaries = [H("s1")]                 # 10
    tail = [H("h1")]                      # 10 (also the current turn)
    briefs = [_brief("c1", 0)]            # >0
    # ceiling fits tail(10)+summaries(10) exactly → briefs get 0 budget and shed.
    kept_s, kept_t, kept_b = _fit_to_ceiling(summaries, tail, briefs, ceiling=20, counter=FakeCounter())
    assert ids(kept_t) == ["h1"]   # tail protected
    assert ids(kept_s) == ["s1"]   # summaries survive
    assert kept_b == []            # briefs shed first


def test_build_view_ceiling_sheds_the_manifest_lead():
    msgs = [H("h1", 10)]
    summaries = [H("s1", 10)]
    tool_calls = {"c1": _brief("c1", 0)}
    view = build_view(msgs, summaries, tool_calls, None, counter=FakeCounter(), view_token_ceiling=20)
    assert ids(view) == ["s1", "h1"]  # no leading manifest — it was shed to fit


# ── state reducers & field isolation ─────────────────────────────────────────

def test_update_dict_adds_without_mutating_existing():
    existing = {"a": 1}
    out = update_dict(existing, {"b": 2})
    assert out == {"a": 1, "b": 2}
    assert existing == {"a": 1}   # input not mutated


def test_update_dict_from_empty_returns_incoming():
    assert update_dict({}, {"a": 1}) == {"a": 1}


def test_context_state_fields_mutable_defaults_are_per_instance():
    s1, s2 = ContextStateFields(), ContextStateFields()
    s1.tool_calls["c1"] = _brief("c1", 0)
    s1.message_summaries.append(H("x"))
    assert s2.tool_calls == {}          # not shared with s1
    assert s2.message_summaries == []


# ── LLMSummarizer (the injected adapter) ─────────────────────────────────────

def test_llm_summarizer_returns_a_single_string_list_and_sends_a_system_prompt():
    model = FakeModel("SUM")
    out = LLMSummarizer(model, "sys-prompt").summarize([H("h1")])
    assert out == ["SUM"]                          # wraps model output as list[str]
    assert isinstance(model.seen[0], SystemMessage)  # summarize, not continue: system-framed


# ── prepare: tool briefing + ordinal seeding (integration) ───────────────────

def _slice_with_a_tool_pair():
    return [H("h1"), Atc("a1", "c1"), T("t1", "c1", n=5), A("a2"), H("hot")]


def test_prepare_briefs_the_tools_in_the_summarized_slice():
    out = _manager().prepare(FakeState(_slice_with_a_tool_pair()), "oceans_agent", FakeSummarizer(["S"]))
    assert "c1" in out["tool_calls"]
    assert out["tool_calls"]["c1"].summary == "rrrrr"
    assert out["tool_calls"]["c1"].ordinal == 0     # seeded from an empty state


def test_prepare_seeds_ordinals_from_existing_tool_calls():
    prior = {"x": _brief("x", 0), "y": _brief("y", 1)}
    state = FakeState(_slice_with_a_tool_pair(), tool_calls=prior)
    out = _manager().prepare(state, "oceans_agent", FakeSummarizer(["S"]))
    assert out["tool_calls"]["c1"].ordinal == 2     # len(prior) == 2 → strictly newer


# ── ContextManager.build_view proxy (threads counter + policy ceiling) ────────

def test_manager_build_view_applies_the_policy_ceiling():
    mgr = ContextManager(FakeCounter(), {"k": ContextPolicy(view_token_ceiling=20)})
    state = FakeState([H("h1", 10)], message_summaries=[H("s1", 10)], tool_calls={"c1": _brief("c1", 0)})
    view = mgr.build_view(state, "k")
    assert ids(view) == ["s1", "h1"]  # ceiling threaded → manifest shed


def test_manager_build_view_without_policy_key_leaves_fitting_off():
    mgr = ContextManager(FakeCounter(), {"k": ContextPolicy(view_token_ceiling=1)})
    state = FakeState([H("h1", 10)], tool_calls={"c1": _brief("c1", 0)})
    view = mgr.build_view(state)      # no policy_key ⇒ ceiling None ⇒ no fitting
    assert isinstance(view[0], HumanMessage) and "Tools already called" in view[0].content
