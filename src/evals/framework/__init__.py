"""evals.framework — agents-agnostic evaluation protocols.

core.py + evaluators.py are pure local code (no langsmith). Task and
Evaluator are what you implement per agents; DatasetSource is how you load
golden cases. Two orchestrators consume the same protocol objects:

    local_runner.py      — in-process loop, scores natively, no langsmith
    langsmith_adapter.py — translates the protocols to LangSmith conventions
                           (the ONLY file that imports langsmith)

    local:     DatasetSource ─▶ run_local_eval() ─▶ Evaluator.evaluate()
    langsmith: DatasetSource ─▶ seed_dataset()   ─▶ LangSmith Dataset
               Task          ─▶ make_target()    ─▶ LangSmith target fn
               Evaluator     ─▶ make_evaluator() ─▶ LangSmith evaluator fn
"""

from evals.framework.core import (
    Case,
    CaseExecution,
    DatasetSource,
    Evaluator,
    Sample,
    Score,
    Task,
    Usage,
)

__all__ = [
    "Case",
    "CaseExecution",
    "DatasetSource",
    "Evaluator",
    "Sample",
    "Score",
    "Task",
    "Usage",
]
