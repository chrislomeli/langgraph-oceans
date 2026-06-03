"""evals.framework.local_runner — run an eval entirely in-process, no LangSmith.

The mirror image of langsmith_adapter.run_langsmith_eval: same Task, same
Evaluator objects, a different orchestrator. This file imports nothing from
langsmith — that absence is the test that the core/adapter seam is drawn in
the right place.

Unlike the LangSmith path (which calls each evaluator once per run and lets
LangSmith aggregate across repetitions), this runner assembles a full
CaseExecution with every sample and hands it to the evaluator in one call —
so in-evaluator reduction (BooleanVote majority, NumericTolerance mean) runs
as designed.

    for case in dataset:            # golden cases
        for _ in range(repeats):    # stochastic samples
            task.run(case.input)
        evaluator.evaluate(execution)   # raw protocol — no wrapper
"""

from __future__ import annotations

import asyncio
import time

from evals.framework.core import (
    Case,
    CaseExecution,
    DatasetSource,
    Evaluator,
    Sample,
    Score,
    Task,
)


async def _run_async(
        task: Task,
        evaluators: list[Evaluator],
        dataset: DatasetSource,
        *,
        repeats: int,
        limit: int,
) -> list[tuple[Case, CaseExecution, list[Score]]]:
    results: list[tuple[Case, CaseExecution, list[Score]]] = []
    for i, case in enumerate(dataset.load()):
        if limit and i >= limit:
            break

        samples: list[Sample] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            output, usage = await task.run(case.input)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            samples.append(Sample(output=output, latency_ms=latency_ms, usage=usage))

        execution = CaseExecution(case=case, samples=samples)
        scores = [s for ev in evaluators for s in ev.evaluate(execution)]
        results.append((case, execution, scores))

    return results


def _report(results: list[tuple[Case, CaseExecution, list[Score]]]) -> None:
    for case, execution, scores in results:
        tokens = sum(s.usage.total_tokens for s in execution.samples)
        print(f"\n===== case: {case.id}  samples={len(execution.samples)}  tokens={tokens} =====")
        if execution.parse_failures:
            print(f"  parse failures: {execution.parse_failures}/{len(execution.samples)}")
        if not scores:
            print("  (no scores emitted)")
        for s in scores:
            gate = "" if s.passed is None else ("PASS" if s.passed else "FAIL")
            print(f"  [{gate:4}] {s.name}={s.value:.2f}  {s.detail}")


def run_local_eval(
        *,
        task: Task,
        evaluators: list[Evaluator],
        dataset: DatasetSource,
        repeats: int = 1,
        limit: int = 0,
) -> list[tuple[Case, CaseExecution, list[Score]]]:
    """Run the dataset through the Task and score it with the Evaluators, in-process.

    Drives the async Task in a single event loop. ``repeats`` controls how
    many samples per case (feeds in-evaluator reduction); ``limit`` caps the
    number of cases (0 = all). Returns per-case (case, execution, scores) and
    prints a summary.
    """
    results = asyncio.run(_run_async(task, evaluators, dataset, repeats=repeats, limit=limit))
    _report(results)
    return results
