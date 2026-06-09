"""trace_one.py — watch ONE eval case go through the whole pipeline, stage by stage.

A teaching script (not part of the eval). It prints the real intermediate values
at every step so the abstract flow becomes concrete:

    frozen split row  →  held-out vector  →  raw nearest images  →
    aggregate to individuals  →  grade (Recall@1 / Recall@5 / MRR)

Run:  uv run python -m evals.photo_id.trace_one            # a default whale
      uv run python -m evals.photo_id.trace_one 6          # whale individual_id=6
"""

from __future__ import annotations

import sys

from stores.postgres import get_pg_gateway
from tools.photo_id import PhotoIDTool, _search_nearest_images


def main() -> None:
    gw = get_pg_gateway()
    tool = PhotoIDTool()
    ver = tool.ver  # the active embedder version (from Settings) drives every query below

    # ── STEP 0: the case, read from the FROZEN split table ────────────────────
    # This is what dataset.py turns into a Case. We did NOT choose it now — it was
    # frozen once by build_split.sql. (Optionally override the whale on the CLI.)
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 19857
    where = "WHERE sp.individual_id = %s" if want else ""
    params = (want,) if want else ()
    row = gw.fetch_rows(
        f"""
        SELECT sp.individual_id, sp.query_sighting_id, i.name
        FROM photo_id_eval_split sp JOIN individuals i USING (individual_id)
        {where}
        ORDER BY sp.individual_id LIMIT 1
        """,
        params,
    )[0]
    true_id, held_out_sighting, name = row["individual_id"], row["query_sighting_id"], row["name"]
    print("STEP 0 — the case (from the frozen split table)")
    print(f"  true whale : individual {true_id}  ({name!r})")
    print(f"  held-out   : sighting {held_out_sighting}  ← we hide THIS photo and must re-find the whale")

    # ── STEP 1: the gallery — this whale's OTHER photos (what's left to match) ─
    gallery = gw.fetch_rows(
        """SELECT count(*) AS n FROM fluke_embeddings
           WHERE individual_id = %s AND embedder_ver = %s 
             AND sighting_id <> %s""",
        (true_id, ver, held_out_sighting),
    )[0]["n"]
    print(f"\nSTEP 1 — the gallery: {gallery} OTHER photos of this whale remain in the catalog")
    print("  (a fair test: the whale IS still findable — just not via the hidden photo)")

    # ── STEP 2: the held-out photo's VECTOR (already computed; stored in the DB) ─
    # No image bytes, no CLIP run — embedding happened back at build time. We just
    # read the 512-number fingerprint. This is task.py's _fetch_vector().
    qvec = gw.fetch_rows(
        "SELECT embedding FROM fluke_embeddings WHERE sighting_id = %s AND embedder_ver = %s",
        (held_out_sighting, ver),
    )[0]["embedding"]
    print(f"\nSTEP 2 — the held-out photo as a vector: {len(qvec)} numbers (CLIP's 'fingerprint')")
    print(f"  first 5 of 512: {[round(float(x), 3) for x in qvec[:5]]}")

    no_holds = _search_nearest_images(gw, qvec, ver, n=12)


    # ── STEP 3: nearest IMAGES in the catalog (raw cosine kNN, hidden one excluded) ─
    # This is the actual vector search (photo_id._search_nearest_images). It returns
    # individual images ranked by similarity. Watch: are the top hits the SAME whale?
    rows = _search_nearest_images(gw, qvec, ver, n=12, exclude_sighting_id=held_out_sighting)
    print("\nSTEP 3 — the 12 nearest IMAGES by cosine similarity (the hidden photo excluded):")
    print("   rank  score   individual   same whale?")
    for i, r in enumerate(rows, 1):
        mark = "  ← TRUE WHALE" if r["individual_id"] == true_id else ""
        print(f"   {i:>4}  {r['score']:.3f}   {r['individual_id']:>8}{mark}")
    print("  Note how scores cluster near ~0.95 across DIFFERENT whales — raw CLIP barely")
    print("  separates individuals. THAT is what the LoRA step later has to fix.")

    # ── STEP 4: collapse images → ranked INDIVIDUALS (the tool's real output) ──
    # query_by_vector aggregates the per-image hits into one row per whale (best
    # photo wins) → the Candidate list the agent/eval actually sees.
    result = tool.query_by_vector(qvec, k=5, exclude_sighting_id=held_out_sighting)
    print("\nSTEP 4 — collapsed to ranked INDIVIDUALS (top 5 candidates):")
    for i, c in enumerate(result.candidates, 1):
        mark = "  ← TRUE WHALE" if c.individual_id == true_id else ""
        print(f"   {i}. individual {c.individual_id:>8} {c.name!r:30} score={c.score:.3f}{mark}")

    # ── STEP 5: the grade (what eval.py's RetrievalRanking computes) ──────────
    no_rank = next((i for i, c in enumerate(result.candidates, 1) if c.individual_id != true_id), None)

    rank = next((i for i, c in enumerate(result.candidates, 1) if c.individual_id == true_id), None)
    print("\nSTEP 5 — the grade for this one case:")
    print(f"  true whale's rank among candidates: {rank if rank else 'not in top 5'}")
    print(f"  reid@1 (top-1 correct?) : {1.0 if rank == 1 else 0.0}")
    print(f"  reid@5 (top-5 correct?) : {1.0 if rank and rank <= 5 else 0.0}")
    print(f"  MRR    (1/rank)         : {1.0 / rank if rank else 0.0:.3f}")
    print("\n  The full baseline is just this, averaged over all 2,486 held-out whales.")


if __name__ == "__main__":
    main()
