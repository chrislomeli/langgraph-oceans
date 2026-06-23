"""evals.framework.core — agents-agnostic evaluation protocols and types.

The design language for all evals in this project. Deliberately stdlib-only
and import-free from the rest of the codebase — the spine never knows what
it is evaluating.

LangSmith handles the run loop, storage, aggregation, and regression
tracking. These protocols are the seam: Task and Evaluator are what you
implement per agents; DatasetSource is how you load golden cases. The
LangSmith adapter (langsmith_adapter.py) translates them to LangSmith's
calling conventions.

    DatasetSource ──▶ seed_dataset() ──▶ LangSmith Dataset
    Task          ──▶ make_target()  ──▶ LangSmith target fn
    Evaluator     ──▶ make_evaluator()──▶ LangSmith evaluator fn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")
Expected = TypeVar("Expected")


# ── Dataset ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case(Generic[Input, Expected]):
    """One golden example: input to feed the SUT and expectation to score against."""

    id: str
    input: Input
    expected: Expected
    tags: tuple[str, ...] = ()
    notes: str = ""


class DatasetSource(Protocol[Input, Expected]):
    """Where golden cases come from — a code list, DB rows, a JSON file.

    ``version`` is recorded on every LangSmith run so results say which
    dataset version they scored.
    """

    version: str

    def load(self) -> list[Case[Input, Expected]]: ...


# ── Cost tracking ─────────────────────────────────────────────────────────────


@dataclass
class Usage:
    """Per-call token/cost tracking."""

    total_tokens: int = 0


# ── Task: the system-under-test seam ─────────────────────────────────────────


class Task(Protocol[Input, Output]):
    """The system under test behind one async call.

    One implementation per agents. ``output`` is None when the LLM call
    failed to produce a parseable result — a scored failure mode, not an
    exception.
    """

    label: str  # recorded in LangSmith as the experiment identifier

    async def run(self, task_input: Input) -> tuple[Output | None, Usage]: ...


# ── Execution records ─────────────────────────────────────────────────────────


@dataclass
class Sample(Generic[Output]):
    """One execution of the Task for a case."""

    output: Output | None
    latency_ms: float
    usage: Usage


@dataclass
class CaseExecution(Generic[Input, Output, Expected]):
    """A case plus every sample the Task produced for it."""

    case: Case[Input, Expected]
    samples: list[Sample[Output]] = field(default_factory=list)

    @property
    def outputs(self) -> list[Output]:
        return [s.output for s in self.samples if s.output is not None]

    @property
    def parse_failures(self) -> int:
        return sum(1 for s in self.samples if s.output is None)


# ── Scores and evaluators ─────────────────────────────────────────────────────


@dataclass
class Score:
    """One scored dimension for one case.

    ``passed`` is the gate: True/False for an assertion, None for an
    informational metric that feeds aggregation but never fails a case.
    """

    name: str
    value: float
    passed: bool | None = None
    detail: str = ""


class Evaluator(Protocol[Output, Expected]):
    """Scores a CaseExecution against the expectation.

    The evaluator owns sample reduction (majority vote, mean, etc.) because
    how you collapse stochastic samples is metric-specific knowledge.
    """

    name: str

    def evaluate(self, execution: CaseExecution[Any, Output, Expected]) -> list[Score]: ...
