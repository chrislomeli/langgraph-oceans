"""evals.escalation.judge — LLM-as-judge factory for the escalation eval.

Returns a judge callable: (reference: str, actual: str) -> (score, reason)
suitable for injection into ReferenceJudge.

The callable is synchronous (LangSmith's evaluator contract is sync).
LangSmith calls each evaluator independently after the target run completes,
so the judge fires once per sample per case — not in the hot path.

The judge returns BOTH a score and a one-line reason. The reason is the
"what was missing / what satisfied it" explanation — ReferenceJudge routes it
into Score.detail, which the LangSmith adapter forwards as the feedback
comment. So a sub-1.0 score is no longer a bare number; you can read why.

Usage:
    from evals.framework.judge import make_llm_judge

    judge = make_llm_judge(llm, system_prompt=_JUDGE_SYSTEM)
    ReferenceJudge(name="reasoning_quality", ..., judge=judge)
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Generic default — domain-neutral framing. Pass system_prompt= to make_llm_judge
# when the eval domain benefits from more specific context for the judge.
_DEFAULT_SYSTEM = """\
You are an impartial evaluator scoring an AI agent's output against provided criteria.
You will be given evaluation criteria and the agent's output to score.
Score it from 0.0 to 1.0:
  0.0 = output does not meet the criteria at all
  0.5 = output partially meets the criteria
  1.0 = output fully meets the criteria
Always give a brief reason. When the score is below 1.0, state specifically
what the output was missing or got wrong relative to the criteria.\
"""

_TEMPLATE = """\
Criteria:
{reference}

Agent output to evaluate:
{actual}\
"""


class JudgeVerdict(BaseModel):
    """Structured judge output — a score plus the reasoning behind it."""

    score: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = does not meet criteria, 0.5 = partial, 1.0 = fully meets.",
    )
    reason: str = Field(
        description="One or two sentences. If score < 1.0, name specifically what was "
        "missing or off relative to the criteria; if 1.0, name what satisfied it.",
    )


def make_llm_judge(llm, *, system_prompt: str = _DEFAULT_SYSTEM):
    """Return a judge callable backed by `llm`.

    Parameters
    ----------
    llm:
        Any LangChain chat model. Invoked synchronously via structured output.
    system_prompt:
        Framing shown to the judge before the criteria and output. Override
        this per eval domain so the judge understands what it is scoring —
        e.g. fire-risk reasoning, logistics plan quality, forecast accuracy.

    The returned callable is: (reference: str, actual: str) -> (float, str)
    where the str is a one-line reason for the score.

    Cases where `reference` is empty string are skipped (returns (0.0, "") with
    no LLM call) — this lets you add ReferenceJudge to the evaluators list
    once and it quietly no-ops on cases that have no criteria authored yet.
    """
    structured = llm.with_structured_output(JudgeVerdict)

    def judge(reference: str, actual: str) -> tuple[float, str]:
        if not reference:
            return 0.0, ""
        prompt = _TEMPLATE.format(reference=reference, actual=actual)
        verdict: JudgeVerdict = structured.invoke(
            [SystemMessage(system_prompt), HumanMessage(prompt)]
        )
        score = min(1.0, max(0.0, float(verdict.score)))
        return score, verdict.reason

    return judge
