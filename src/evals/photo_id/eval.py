"""eval.py — run the Layer-A photo-ID baseline (roadmap item 6).

Scores closed-set leave-one-out re-identification with the framework's
`RetrievalRanking` reused VERBATIM: each candidate individual becomes a Span keyed
on its id, the truth is the held-out individual's id, and Span equality reduces to
"same whale?". Headline metrics: Recall@1 (top-1 accuracy), Recall@5, MRR.

    # local aggregate (no LangSmith account needed) — see the baseline number now:
    uv run python -m evals.photo_id.eval
    uv run python -m evals.photo_id.eval --limit 200        # quick look

    # by-individual GENERALIZATION eval (reid_split; the honest LoRA yardstick):
    uv run python -m evals.photo_id.eval --split val            # iterate here
    uv run python -m evals.photo_id.eval --split test --final   # spend ONCE at the end

    # push the durable baseline experiment to LangSmith:
    uv run python -m evals.photo_id.eval --langsmith
    uv run python -m evals.photo_id.eval --langsmith --seed-only   # seed dataset only

Precision@k is intentionally NOT a headline: with exactly one correct individual,
precision@5 caps at 0.2 even on a perfect hit. Recall@k (= top-k accuracy) and MRR
are the re-ID metrics. We EXPECT raw CLIP to score low — this run is the fixed
baseline the Phase-C LoRA lift is measured against.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict
from statistics import mean

from evals.framework.core import CaseExecution, Sample
from evals.framework.evaluators import RetrievalRanking, Span
from evals.photo_id.dataset import PhotoIDLeaveOneOut
from evals.photo_id.reid_dataset import ReIDSplitDataset
from evals.photo_id.task import PhotoIDTask, ReIDTask

log = logging.getLogger(__name__)


def build_evaluators() -> list[RetrievalRanking]:
    """RetrievalRanking reused for re-ID: Span(path=str(individual_id)) ⇒ id-equality."""
    ranked = lambda out: [Span(path=str(c.individual_id)) for c in out.candidates]  # noqa: E731
    relevant = lambda case: [Span(path=str(case.expected["individual_id"]))]  # noqa: E731
    return [
        RetrievalRanking(name="reid@1", ranked=ranked, relevant=relevant, k=1),
        RetrievalRanking(name="reid@5", ranked=ranked, relevant=relevant, k=5),
    ]


async def _local_baseline(task, evaluators, dataset, limit: int) -> None:
    """Aggregate Recall@k / MRR across all cases — the baseline headline, no per-case spam."""
    cases = dataset.load()
    if limit:
        cases = cases[:limit]
    agg: dict[str, list[float]] = defaultdict(list)
    misses = 0
    for i, case in enumerate(cases, 1):
        output, usage = await task.run(case.input)
        execution = CaseExecution(case=case, samples=[Sample(output=output, latency_ms=0.0, usage=usage)])
        for ev in evaluators:
            for s in ev.evaluate(execution):
                agg[s.name].append(s.value)
        if output is None or not output.candidates:
            misses += 1
        if i % 500 == 0:
            log.info("scored %d/%d", i, len(cases))

    n = len(cases)
    print(f"\n===== photo-id Layer-A baseline ({task.label}) =====")
    print(f"cases: {n}   (held-out individuals, closed-set leave-one-out)")
    for metric in ("reid@1", "reid@5", "reid@5_mrr"):
        vals = agg.get(metric)
        if vals:
            print(f"  {metric:12s} = {mean(vals):.3f}")
    print(f"  empty/parse-fail: {misses}")
    print("  (reid@1 = top-1 accuracy, reid@5 = top-5 accuracy, mrr = mean reciprocal rank)")


def _verify_feedback(client, dataset_id: str, expect_runs: int) -> None:
    """After a LangSmith run, confirm feedback actually landed (the flush check).

    Polls briefly for the newest experiment on the dataset, then reports how many
    runs carry feedback. Small datasets index in seconds, so this cleanly tells
    'fix works' (all runs have feedback) from 'slow' (runs not visible yet).
    """
    import time
    from collections import Counter

    for _ in range(12):  # ~60s
        projs = list(client.list_projects(reference_dataset_id=dataset_id))
        newest = max(projs, key=lambda p: str(p.start_time or "")) if projs else None
        runs = list(client.list_runs(project_id=newest.id, is_root=True)) if newest else []
        if len(runs) >= expect_runs:
            fb = list(client.list_feedback(run_ids=[r.id for r in runs]))
            with_fb = len({f.run_id for f in fb})
            print(f"\nVERIFY '{newest.name}': {len(runs)} runs, {with_fb} carry feedback")
            print(f"  feedback keys: {dict(Counter(f.key for f in fb))}")
            print("  ✅ FIX CONFIRMED — feedback flushed" if with_fb == len(runs)
                  else "  ⚠️  feedback still missing on some runs")
            return
        time.sleep(5)
    print("\nVERIFY: runs not indexed yet (ingestion lag) — check the dashboard shortly")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Layer-A photo-ID retrieval baseline")
    ap.add_argument("--langsmith", action="store_true", help="seed + run via LangSmith (durable baseline)")
    ap.add_argument("--seed-only", action="store_true", help="with --langsmith: seed the dataset, skip the run")
    ap.add_argument("--smoke", type=int, default=0, help="tiny N-case self-verifying LangSmith smoke test")
    ap.add_argument("--verify-only", action="store_true",
                    help="don't run; just check the newest experiment's feedback (use after ingestion lag clears)")
    ap.add_argument("--limit", type=int, default=0, help="cap cases for a LOCAL run (0 = all)")
    ap.add_argument("--k", type=int, default=10, help="candidates returned per query")
    ap.add_argument("--split", choices=["val", "test"], default=None,
                    help="run the by-individual reid eval on this reid_split split "
                         "(omit for the original closed-set leave-one-out)")
    ap.add_argument("--final", action="store_true",
                    help="required with --split test — the test set is spent ONCE, at the end")
    args = ap.parse_args()

    if args.split == "test" and not args.final:
        ap.error("--split test requires --final: iterate on --split val; "
                 "test is scored once, when the model is frozen.")

    # --smoke N: a small, isolated dataset run through LangSmith, then verified.
    smoke = args.smoke > 0
    if args.split:
        dataset = ReIDSplitDataset(split=args.split, limit=args.smoke)
        task = ReIDTask(gallery_individual_ids=dataset.gallery_individual_ids(), k=args.k)
    else:
        dataset = PhotoIDLeaveOneOut(limit=args.smoke) if smoke else PhotoIDLeaveOneOut()
        task = PhotoIDTask(k=args.k)
    evaluators = build_evaluators()

    if args.langsmith or smoke:
        from langsmith import Client

        from config import get_settings
        from evals.framework.langsmith_adapter import run_langsmith_eval, seed_dataset
        from tools.photo_id import PhotoIDResult

        # Export LangSmith creds from settings (.env) into the env the SDK reads —
        # the caller owns settings setup (harness convention).
        get_settings().apply_langsmith()
        client = Client()

        # --verify-only: skip running; re-check the newest experiment's feedback once
        # ingestion lag has cleared (the reliable read-back the live run can't do).
        if args.verify_only:
            ds = client.read_dataset(dataset_name=dataset.name)
            _verify_feedback(client, str(ds.id), expect_runs=args.smoke or 1)
            return

        dataset_id = seed_dataset(dataset, client=client, dataset_name=dataset.name)
        print(f"Dataset '{dataset.name}' ready (id={dataset_id})")
        if args.seed_only:
            print("--seed-only: skipping run.")
            return
        prefix = f"{'smoke' if smoke else 'baseline'}-{task.label}"
        run_langsmith_eval(
            task=task,
            evaluators=evaluators,
            dataset_name=dataset.name,
            experiment_prefix=prefix,
            num_repetitions=1,  # retrieval is deterministic — no REPEATS needed (unlike the agents)
            output_model=PhotoIDResult,
            client=client,  # same client we flush, so feedback fully drains
        )
        if smoke:
            _verify_feedback(client, dataset_id, expect_runs=args.smoke)
        return

    asyncio.run(_local_baseline(task, evaluators, dataset, args.limit))


if __name__ == "__main__":
    main()
