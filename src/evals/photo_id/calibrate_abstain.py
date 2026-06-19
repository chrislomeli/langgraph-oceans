"""calibrate_abstain.py — set photo_id's ABSTAIN_THRESHOLD from VAL data (roadmap B0).

`abstain` is an OPEN-SET detector: "is this whale in the catalog at all?" — NOT a
check on whether the top-1 ranking is the correct individual. So we calibrate the
top-1-score cutoff τ that separates two populations, both drawn from the reid_split
VAL whales (test is never touched):

  • genuine  — the true individual IS in the gallery (closed-set). Should NOT abstain.
  • impostor — the true individual is ABSENT (a stand-in for a NOVEL animal).
               Should abstain.

The impostor population is built for free, no new data: for each val whale we run ONE
leave-one-out retrieval against the full val gallery, then read two scores off it —
  genuine top-1  = candidates[0].score          (true whale eligible)
  impostor top-1 = first candidate that is NOT the true individual
                                                (true whale removed → a novel animal)
Same retrieval, perfectly paired. When the true whale doesn't even rank top-k the two
scores coincide (it was effectively absent already), which is correct.

τ is chosen by Youden's J = TPR − FPR, maximized over the observed scores:
  TPR = P(score ≥ τ | genuine)    1−TPR = known whales wrongly called NOVEL
  FPR = P(score ≥ τ | impostor)     FPR  = novel animals wrongly called a match

    uv run python -m evals.photo_id.calibrate_abstain            # all val whales
    uv run python -m evals.photo_id.calibrate_abstain --limit 200  # quick look
"""

from __future__ import annotations

import argparse
import logging
from statistics import mean, median

from evals.photo_id.reid_dataset import ReIDSplitDataset
from tools.photo_id import PhotoIDTool

log = logging.getLogger(__name__)


def _pct(xs: list[float], p: float) -> float:
    """The p-th percentile (0..100) by nearest-rank — stdlib only."""
    s = sorted(xs)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


def _summary(label: str, xs: list[float]) -> None:
    print(
        f"  {label:9s} n={len(xs):4d}  "
        f"min={min(xs):.3f}  p10={_pct(xs, 10):.3f}  median={median(xs):.3f}  "
        f"mean={mean(xs):.3f}  p90={_pct(xs, 90):.3f}  max={max(xs):.3f}"
    )


def collect_scores(limit: int) -> tuple[list[float], list[float]]:
    """Return (genuine_top1_scores, impostor_top1_scores) over val whales."""
    ds = ReIDSplitDataset(split="val")
    gallery = ds.gallery_individual_ids()
    cases = ds.load()
    if limit:
        cases = cases[:limit]

    tool = PhotoIDTool()  # uses the active embedder_ver (v3)
    genuine: list[float] = []
    impostor: list[float] = []

    for i, case in enumerate(cases, 1):
        sid = case.input["query_sighting_id"]
        true_id = case.input["individual_id"]
        rows = tool.gw.fetch_rows(
            "SELECT embedding FROM fluke_embeddings WHERE sighting_id = %s AND embedder_ver = %s",
            (sid, tool.ver),
        )
        if not rows:
            continue
        qvec = rows[0]["embedding"]
        # One retrieval against the whole val gallery, leave-one-out. k large enough
        # that a #2 always exists after we drop the true individual.
        res = tool.query_by_vector(
            qvec, k=25, exclude_sighting_id=sid, restrict_individual_ids=gallery
        )
        if not res.candidates:
            continue
        genuine.append(res.candidates[0].score)
        impostor_cands = [c for c in res.candidates if c.individual_id != true_id]
        if impostor_cands:
            impostor.append(impostor_cands[0].score)
        if i % 500 == 0:
            log.info("scored %d/%d", i, len(cases))

    return genuine, impostor


def sweep(genuine: list[float], impostor: list[float]) -> None:
    """Print a TPR/FPR/accuracy table over candidate τ and the Youden's-J pick."""
    ng, ni = len(genuine), len(impostor)

    def rates(tau: float) -> tuple[float, float, float]:
        tpr = sum(s >= tau for s in genuine) / ng  # genuine kept as a match
        fpr = sum(s >= tau for s in impostor) / ni  # impostor wrongly kept
        acc = (sum(s >= tau for s in genuine) + sum(s < tau for s in impostor)) / (ng + ni)
        return tpr, fpr, acc

    # Candidate τ = every observed score (the optimum always sits at one of these).
    candidates = sorted(set(genuine) | set(impostor))
    best_tau, best_j = candidates[0], -1.0
    for tau in candidates:
        tpr, fpr, _ = rates(tau)
        if tpr - fpr > best_j:
            best_j, best_tau = tpr - fpr, tau

    print("\n  τ      TPR    FPR    accuracy   (TPR=keep known, FPR=keep novel)")
    grid = [round(x, 2) for x in _frange(0.30, 0.75, 0.05)] + [round(best_tau, 4)]
    for tau in sorted(set(grid)):
        tpr, fpr, acc = rates(tau)
        mark = "  ← Youden's J" if abs(tau - best_tau) < 1e-9 else ""
        print(f"  {tau:.3f}  {tpr:.3f}  {fpr:.3f}  {acc:.3f}{mark}")

    tpr, fpr, acc = rates(best_tau)
    print(f"\n  CHOSEN τ = {best_tau:.4f}  (Youden's J = {best_j:.3f})")
    print(f"    • {(1 - tpr) * 100:.1f}% of KNOWN whales would be wrongly called NOVEL (false abstain)")
    print(f"    • {fpr * 100:.1f}% of NOVEL animals would be wrongly called a match (false match)")
    print(f"    • balanced accuracy on the open-set decision: {acc:.3f}")


def _frange(lo: float, hi: float, step: float):
    x = lo
    while x <= hi + 1e-9:
        yield x
        x += step


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Calibrate photo_id ABSTAIN_THRESHOLD on val")
    ap.add_argument("--limit", type=int, default=0, help="cap val whales (0 = all)")
    args = ap.parse_args()

    genuine, impostor = collect_scores(args.limit)
    print("\n===== abstain calibration — reid_split VAL (top-1 cosine) =====")
    _summary("genuine", genuine)
    _summary("impostor", impostor)
    sweep(genuine, impostor)
    print("\n(Calibrated on VAL only — test stays sealed. Apply τ to tools/photo_id.py next.)")


if __name__ == "__main__":
    main()
