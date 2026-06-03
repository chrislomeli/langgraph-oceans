"""evals.framework.harness — one entry point for running an eval either way.

Both orchestrators (local_runner, langsmith_adapter) consume the same Task /
Evaluator / DatasetSource objects; this module owns the boilerplate that used
to be copy-pasted into every eval's __main__: the local/langsmith branch,
usage-counter reset, dataset seeding, and the per-role cost report.

Each eval is then just: build task + evaluators + dataset, call run_eval().

Seam note
─────────
langsmith is imported lazily, only inside the langsmith branch. A local run
(`langsmith=False`) imports nothing from langsmith — so the offline path
works with no LangSmith account or package.
"""

from __future__ import annotations

from evals.framework.core import DatasetSource, Evaluator, Task
from evals.framework.local_runner import run_local_eval


def _print_usage(llm_registry) -> None:
    """Per-role token/cost report — the tail every eval printed by hand."""
    print("Done.")
    for row in llm_registry.usage_report():
        cost = row["estimated_cost_usd"]
        cost_str = f"  cost=${cost:.4f}" if cost is not None else ""
        print(
            f"  [{row['role']}] calls={row['calls']}  tokens={row['total_tokens']}"
            f" (in={row['input_tokens']} out={row['output_tokens']}){cost_str}"
        )


def run_eval(
        *,
        task: Task,
        evaluators: list[Evaluator],
        dataset: DatasetSource,
        llm_registry,
        langsmith: bool,
        experiment_prefix: str,
        repeats: int = 3,
        input_model=None,
        output_model=None,
        limit: int = 0,
        seed_only: bool = False,
):
    """Run an eval through either orchestrator and print the usage report.

    langsmith=True  → seed the dataset (idempotent) and run via LangSmith;
                      results live in the dashboard. seed_only stops here.
    langsmith=False → run in-process via local_runner; per-case scores print
                      to stdout. seed_only is ignored (nothing to seed).

    The caller owns settings/registry/prompt setup; this owns reset + branch
    + report so all evals behave identically.
    """
    # Zero counters so the report reflects only this run, not setup tokens
    # (judge construction, etc.).
    llm_registry.reset_usage()

    try:
        if langsmith:
            # Lazy: keep the offline path free of any langsmith import.
            from langsmith import Client

            from evals.framework.langsmith_adapter import run_langsmith_eval, seed_dataset

            client = Client()
            dataset_id = seed_dataset(dataset, client=client, dataset_name=dataset.name)
            print(f"Dataset '{dataset.name}' ready (id={dataset_id})")
            if seed_only:
                print("--seed-only: skipping eval run.")
                return None

            print(f"Running eval: {experiment_prefix} × {repeats} repetitions (LangSmith) …")
            return run_langsmith_eval(
                task=task,
                evaluators=evaluators,
                dataset_name=dataset.name,
                experiment_prefix=experiment_prefix,
                num_repetitions=repeats,
                input_model=input_model,
                output_model=output_model,
            )

        print(f"Running eval: {experiment_prefix} × {repeats} repetitions (local) …")
        return run_local_eval(
            task=task,
            evaluators=evaluators,
            dataset=dataset,
            repeats=repeats,
            limit=limit,
        )
    finally:
        _print_usage(llm_registry)
