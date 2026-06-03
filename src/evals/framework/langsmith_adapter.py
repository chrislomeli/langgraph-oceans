"""evals.framework.langsmith_adapter — the seam between local protocols and LangSmith.

This is the ONLY file in the eval framework that imports langsmith. It
translates the local Task / Evaluator / DatasetSource protocols into
LangSmith's calling conventions so the rest of the framework stays
import-free from any platform. Run a dataset with no LangSmith account via
``local_runner.run_local_eval`` — it uses the same Task and Evaluator
objects, just a different orchestrator.

Operations:

    _serialize / _deserialize — pydantic <-> dict round-trip (LS stores JSON)
    seed_dataset()      — push a DatasetSource to LangSmith once (idempotent)
    make_target()       — wrap a Task as LangSmith's dict→dict target fn
    make_evaluator()    — wrap an Evaluator as LangSmith's (run,example)→list fn
    run_langsmith_eval()— top-level entry point
"""

from __future__ import annotations

import asyncio
from typing import Any

from langsmith import EvaluationResult
from langsmith.evaluation import EvaluationResults

from evals.framework.core import Case, CaseExecution, DatasetSource, Evaluator, Sample, Task, Usage


# ── Serde ───────────────────────────────────────────────────────────────────
# LangSmith stores inputs/outputs as JSON, so models cross the boundary as
# dicts. The local runner needs none of this — it scores native objects.


def _serialize(obj: Any) -> Any:
    """Pydantic models → dict; primitives pass through."""
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def _deserialize(obj: Any, model_class=None):
    """dict → Pydantic model if a class is provided; otherwise return raw."""
    if model_class is not None and isinstance(obj, dict):
        return model_class.model_validate(obj)
    return obj


# ── Target wrapper ────────────────────────────────────────────────────────────


def make_target(task: Task, *, input_model=None):
    """Wrap a Task into LangSmith's target signature: dict → dict.

    LangSmith calls target(example.inputs) and stores the return value as
    run.outputs. The async Task is driven synchronously here; LangSmith
    manages concurrency at the experiment level via max_concurrency.
    """

    async def _run(inputs: dict) -> dict:
        raw = inputs.get("input", inputs)
        task_input = _deserialize(raw, input_model)
        output, usage = await task.run(task_input)
        return {
            "result": _serialize(output),
            "total_tokens": usage.total_tokens,
        }

    def target(inputs: dict) -> dict:
        return asyncio.run(_run(inputs))

    target.__name__ = task.label
    return target


# ── Evaluator wrapper ─────────────────────────────────────────────────────────


def make_evaluator(evaluator: Evaluator, *, output_model=None, expected_key: str = "expected"):
    """Wrap a framework Evaluator into LangSmith's (run, example) → list[dict] signature.

    Reconstructs a minimal CaseExecution from LangSmith's run/example so
    existing Evaluator implementations work unchanged. Note: LangSmith calls
    this once per run, so the CaseExecution carries a single sample — sample
    reduction (majority vote, mean) happens via LangSmith's own aggregation
    across repetitions, not inside the evaluator. The local_runner is the
    path that exercises in-evaluator reduction over multiple samples.
    """

    def ls_evaluator(run, example) -> EvaluationResults:
        raw_output = (run.outputs or {}).get("result")
        raw_expected = (example.outputs or {}).get(expected_key)

        output = _deserialize(raw_output, output_model)
        case = Case(
            id=str(example.id),
            input=example.inputs.get("input"),
            expected=raw_expected,
        )
        execution = CaseExecution(
            case=case,
            samples=[Sample(output=output, latency_ms=0.0, usage=Usage())],
        )

        scores = evaluator.evaluate(execution)
        return EvaluationResults(results=[
            EvaluationResult(key=s.name, score=s.value, comment=s.detail)
            for s in scores
        ])

    ls_evaluator.__name__ = evaluator.name
    return ls_evaluator


# ── Dataset seeding ───────────────────────────────────────────────────────────


def seed_dataset(
        source: DatasetSource,
        *,
        client,
        dataset_name: str,
) -> str:
    """Push a DatasetSource into LangSmith as a named, versioned dataset.

    Idempotent — skips population if the dataset already contains examples.
    Returns the dataset ID.

    To reseed after changing golden cases: delete the dataset in the LangSmith
    UI (or via client.delete_dataset()), then run with --seed-only again.
    """
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if datasets:
        dataset = datasets[0]
        existing = list(client.list_examples(dataset_id=dataset.id, limit=1))
        if existing:
            return str(dataset.id)
    else:
        dataset = client.create_dataset(
            dataset_name,
            description=f"{source.__class__.__name__} v{source.version}",
        )

    cases = source.load()
    inputs = [{"input": _serialize(c.input), "case_id": c.id} for c in cases]
    outputs = [{"expected": _serialize(c.expected), "tags": list(c.tags), "notes": c.notes} for c in cases]
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id,
    )
    return str(dataset.id)


# ── Top-level harness ─────────────────────────────────────────────────────────


def run_langsmith_eval(
        *,
        task: Task,
        evaluators: list[Evaluator],
        dataset_name: str,
        experiment_prefix: str,
        num_repetitions: int = 3,
        max_concurrency: int = 3,
        input_model=None,
        output_model=None,
):
    """Run the eval. LangSmith handles sampling, storage, and the UI.

    Results, per-case scores, and run history are visible in the LangSmith
    dashboard. Regression detection is available via the Experiments view.
    """
    from langsmith import evaluate

    return evaluate(
        make_target(task, input_model=input_model),
        data=dataset_name,
        evaluators=[make_evaluator(e, output_model=output_model) for e in evaluators],
        experiment_prefix=experiment_prefix,
        num_repetitions=num_repetitions,
        max_concurrency=max_concurrency,
    )
