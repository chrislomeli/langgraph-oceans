"""evals.framework.evaluators — reusable, agent-agnostic scoring strategies.

These are the "scoring toolkit" half of an evaluator. Each strategy is generic
and is wired to a specific output via accessor callables you pass in — so you
build the strategies once, and per agent you only choose *which strategy reads
which field*. None of them import anything agent-specific.

    BooleanVote      — majority vote of a boolean decision vs expected (exact)
    NumericTolerance — mean of a number within ±tolerance of expected (banded + MAE)
    KeywordPresence  — cheap, deterministic groundedness: required substrings present
    RetrievalRanking — Precision@k / Recall@k / MRR over ranked spans vs anchors
    ReferenceJudge   — the LLM-as-judge seam, with the judge injected (no LLM dep here)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from evals.framework.core import CaseExecution, Score


def _majority(flags: Sequence[bool]) -> bool:
    return sum(1 for f in flags if f) * 2 > len(flags)


# ── Retrieval geometry ────────────────────────────────────────────────────────
# Both ground-truth anchors and retrieved chunks reduce to the same shape: a
# located code region. Keeping the geometry here (not in any agent) is what lets
# RetrievalRanking stay agent-agnostic — the agent only supplies accessors that
# turn its own chunk type into Spans.


@dataclass(frozen=True)
class Span:
    """A located code region — a path plus an optional symbol and/or line range.

    Used for two roles that happen to share a shape:
      - a ground-truth *anchor* (hand-authored: "the answer lives at this symbol")
      - a retrieved *chunk* (what the retriever actually returned)

    Anchors are keyed on path + symbol on purpose — NOT on chunk ids. Chunk
    boundaries change every time you re-chunk, but "the cosine score lives in
    search_chunks" stays true, so ground truth authored this way survives a
    chunking refactor (which is the very thing the eval exists to measure).
    """

    path: str
    symbol: str | None = None
    start: int | None = None
    end: int | None = None


def _path_suffix_match(a: str, b: str) -> bool:
    """True if one path is a trailing-segment suffix of the other.

    Robust to whether the corpus stores a bare filename or a repo-relative path:
    we compare only the min(len) trailing segments, so 'code_intel_repo.py',
    'postgres/code_intel_repo.py' and 'src/stores/postgres/code_intel_repo.py'
    all match each other.
    """
    pa = [s for s in a.strip("/").split("/") if s]
    pb = [s for s in b.strip("/").split("/") if s]
    n = min(len(pa), len(pb))
    return n > 0 and pa[-n:] == pb[-n:]


def _spans_overlap(a: Span, b: Span) -> bool:
    if a.start is None or a.end is None or b.start is None or b.end is None:
        return False
    return a.start <= b.end and b.start <= a.end


def _hits(anchor: Span, chunk: Span) -> bool:
    """Does a retrieved chunk satisfy a ground-truth anchor?

    Path must match, then ANY of:
      - symbol match  : the anchor names a symbol and the chunk carries it
      - span overlap  : both have line ranges and they intersect
      - file-level    : the anchor names neither symbol nor lines (any chunk
                        from the file counts)
    """
    if not _path_suffix_match(anchor.path, chunk.path):
        return False
    symbol_match = anchor.symbol is not None and anchor.symbol == chunk.symbol
    span_match = anchor.start is not None and _spans_overlap(anchor, chunk)
    file_level = anchor.symbol is None and anchor.start is None
    return symbol_match or span_match or file_level


def _label(span: Span) -> str:
    """Compact human label for a Span: 'file.py:symbol' / 'file.py:10-20' / 'file.py'."""
    name = span.path.rstrip("/").split("/")[-1]
    if span.symbol:
        return f"{name}:{span.symbol}"
    if span.start is not None:
        return f"{name}:{span.start}-{span.end}"
    return name


@dataclass
class BooleanVote:
    """Score a boolean decision by majority vote across samples vs the expected bool.

    predict  : Output   -> bool   (e.g. lambda o: o.escalate)
    expected : Expected -> bool   (e.g. lambda x: x.escalate)
    """

    name: str
    predict: Callable[[Any], bool]
    expected: Callable[[Any], bool]

    def evaluate(self, ex: CaseExecution) -> list[Score]:
        outs = ex.outputs
        if not outs:
            return [Score(self.name, 0.0, passed=False, detail="all samples parse-failed")]
        expected_val = self.expected(ex.case.expected)
        if expected_val is None:
            # Ambiguous case (no ground-truth answer). Emit NO score at all —
            # returning a 0.0 here would be recorded as a failed run downstream
            # and drag the aggregate down. An empty list means "this evaluator
            # has nothing to say about this case", so it's omitted cleanly.
            return []
        actual_outputs = [bool(self.predict(o)) for o in outs]
        actual_majority = _majority(actual_outputs)
        expected = bool(expected_val)
        ok = actual_majority == expected
        return [
            Score(
                self.name,
                float(ok),
                passed=ok,
                detail=f"{sum(actual_outputs)}/{len(actual_outputs)} true; decided={actual_majority} want={expected}",
            )
        ]


@dataclass
class NumericTolerance:
    """Score a numeric field by mean-across-samples within ±tolerance of expected.

    Emits two scores: ``<name>`` (pass/fail within tolerance) and ``<name>_mae``
    (absolute error, informational — feeds a mean/MAE aggregator).
    """

    name: str
    predict: Callable[[Any], float]
    expected: Callable[[Any], float]
    tolerance: float

    def evaluate(self, ex: CaseExecution) -> list[Score]:
        outs = ex.outputs
        if not outs:
            return [Score(self.name, 0.0, passed=False, detail="all samples parse-failed")]
        avg = mean(float(self.predict(o)) for o in outs)
        want = float(self.expected(ex.case.expected))
        err = abs(avg - want)
        ok = err <= self.tolerance
        return [
            Score(
                self.name,
                float(ok),
                passed=ok,
                detail=f"mean={avg:.2f} want={want:.2f} err={err:.2f} tol={self.tolerance}",
            ),
            Score(f"{self.name}_mae", err, passed=None),
        ]


@dataclass
class KeywordPresence:
    """Deterministic groundedness check: all required substrings appear in the text.

    text : Output -> str  (e.g. lambda o: o.temperature_humidity + " " + o.wind)

    Useful where 'fuzzy' is actually checkable without a judge — cheaper and more
    reliable than an LLM when you just need "did it cite the thing it must cite".
    """

    name: str
    text: Callable[[Any], str]
    required: Callable[[Any], Sequence[str]]

    def evaluate(self, ex: CaseExecution) -> list[Score]:
        outs = ex.outputs
        required = self.required(ex.case)

        if not outs or not required:
            return [Score(self.name, 0.0, passed=None, detail="not scored")]

        haystack = " ".join(self.text(o).lower() for o in outs)
        hits = [kw for kw in required if kw.lower() in haystack]

        ok = len(hits) == len(required)
        return [
            Score(
                self.name,
                len(hits) / len(required),
                passed=ok,
                detail=f"{len(hits)}/{len(required)} keywords present",
            )
        ]


@dataclass
class RetrievalRanking:
    """Layer-A retrieval scoring: Precision@k / Recall@k / MRR. Deterministic, no judge.

    Scores the *ranked list the retriever returned* — not the synthesized answer.
    That separation is the whole point: "the model phrased it three valid ways" is
    an answer-quality question (Layer B / ReferenceJudge); here we only ask "did the
    right code show up in the top-k, and how high?", which has a fact of the matter
    against a closed corpus you authored.

      ranked   : Output -> ranked Spans, best first (the retrieved chunks as Spans)
      relevant : Case   -> all ground-truth anchor Spans (graded relevance set)
      must     : Case   -> the subset Recall@k gates on; defaults to `relevant`.
                 Anchors outside `must` are "nice-to-find": they can earn Precision
                 and MRR credit but never fail Recall.

    Emits three scores, mirroring NumericTolerance's gated-plus-companions shape:
      <name>                 Recall@k   (gated: covered must-anchors / total must)
      <name>_precision@<k>    Precision@k (informational)
      <name>_mrr              MRR        (informational)

    Recall is the gate because the question that matters is "did we surface the
    code the answer needs"; precision and rank quality are diagnostics on top.
    Retrieval is deterministic, so the across-sample mean is just defensive.
    """

    name: str
    ranked: Callable[[Any], Sequence[Span]]
    relevant: Callable[[Any], Sequence[Span]]
    must: Callable[[Any], Sequence[Span]] | None = None
    k: int = 5
    recall_threshold: float = 1.0

    def evaluate(self, ex: CaseExecution) -> list[Score]:
        anchors = list(self.relevant(ex.case))
        if not anchors:
            # No retrieval ground truth authored for this case — say nothing,
            # same as BooleanVote's ambiguous branch (a 0.0 would wrongly drag
            # the aggregate down).
            return []
        must = list(self.must(ex.case)) if self.must else anchors

        per_sample: list[tuple[float, float, float]] = []
        for out in ex.outputs:
            topk = list(self.ranked(out))[: self.k]
            hit_flags = [any(_hits(a, c) for a in anchors) for c in topk]

            precision = (sum(hit_flags) / len(topk)) if topk else 0.0
            covered = sum(1 for a in must if any(_hits(a, c) for c in topk)) if must else 0
            recall = covered / len(must) if must else 0.0
            rr = next((1.0 / (i + 1) for i, hit in enumerate(hit_flags) if hit), 0.0)
            per_sample.append((precision, recall, rr))

        if not per_sample:
            return [Score(self.name, 0.0, passed=False, detail="all samples retrieved nothing")]

        p = mean(s[0] for s in per_sample)
        r = mean(s[1] for s in per_sample)
        m = mean(s[2] for s in per_sample)
        ok = r >= self.recall_threshold

        # Triage aid: which must-anchors fell out of the top-k, and what showed up
        # instead. Computed from the first sample (retrieval is deterministic) so a
        # failing case reads like a sorted to-do, not just a number. This is what
        # tells you whether a miss is an unfair anchor, a vector blind spot, or a
        # chunking/ingest problem.
        first = list(self.ranked(ex.outputs[0]))[: self.k]
        missed = [_label(a) for a in must if not any(_hits(a, c) for c in first)]
        got = [_label(c) for c in first]
        detail = f"recall@{self.k}={r:.2f} (thr {self.recall_threshold}); {len(must)} must-find anchor(s)"
        if missed:
            detail += f"; MISSED {missed}; got {got}"

        return [
            Score(
                self.name,
                r,
                passed=ok,
                detail=detail,
            ),
            Score(f"{self.name}_precision@{self.k}", p, passed=None, detail=f"precision@{self.k}={p:.2f}"),
            Score(f"{self.name}_mrr", m, passed=None, detail=f"mrr={m:.2f} (first relevant hit's reciprocal rank)"),
        ]


@dataclass
class ReferenceJudge:
    """The LLM-as-judge seam — with NO LLM dependency baked in.

    You inject ``judge``, a function ``(reference, actual) -> score in [0, 1]``.
    In production that's a rubric-pinned LLM call; in tests it's a stub. Scores
    ≥ ``threshold`` pass. This is where prose / faithfulness scoring plugs in,
    without coupling the framework to any model provider.
    """

    name: str
    reference: Callable[[Any], str]  # Expected -> reference text
    actual: Callable[[Any], str]  # Output   -> produced text
    judge: Callable[[str, str], tuple[float, str]]  # (reference, actual) -> (score, reason)
    threshold: float = 0.7

    def evaluate(self, ex: CaseExecution) -> list[Score]:
        outs = ex.outputs
        if not outs:
            return [Score(self.name, 0.0, passed=False, detail="all samples parse-failed")]
        ref = self.reference(ex.case.expected)
        if not ref:
            # No criteria authored for this case. Emit NO score — same reasoning
            # as BooleanVote's ambiguous branch: a 0.0 here would be recorded as
            # a failed run and drag the aggregate down. Empty list = "nothing to
            # say about this case", so it's omitted cleanly.
            return []
        graded = [self.judge(ref, self.actual(o)) for o in outs]
        scores = [score for score, _ in graded]
        reasons = [reason for _, reason in graded if reason]
        avg = mean(scores)
        ok = avg >= self.threshold
        # Surface the judge's reasoning in detail → flows to the LangSmith
        # feedback comment, so a sub-1.0 score says WHY, not just how much.
        reason_text = " | ".join(reasons) if reasons else "(no reason given)"
        detail = f"mean={avg:.2f} thr={self.threshold} — {reason_text}"
        return [Score(self.name, avg, passed=ok, detail=detail)]
