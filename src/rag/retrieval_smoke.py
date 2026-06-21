"""retrieval_smoke.py — the design-note checkpoint (rag-preprocessing-design.md §95).

Proves the metadata scoping is load-bearing, not decorative, via an A/B on the SAME
query:
    (A) NO species filter  → blue/fin/etc. chunks contaminate a humpback question
                              (textually near-identical SARs — the failure mode).
    (B) species=Humpback   → humpback-only chunks from the Mortality section.

If (A) is mixed and (B) is clean, scoping works. Run:
    uv run python -m rag.retrieval_smoke
"""

from __future__ import annotations

from config import get_settings
from stores.postgres.embedder import Embedder
from stores.postgres.rag_repo import DocumentRepository

QUERY = "humpback whale deaths and serious injury from vessel strikes"


def _show_vec(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    species_seen = set()
    for r in rows:
        sp = ", ".join(r["species"])
        species_seen.add(sp)
        print(f"  {r['distance']:.3f}  {sp:14} | {r['section'][:34]:34} | {r['source']}")
    print(f"  → species in results: {sorted(species_seen)}")


def _show_hybrid(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"  {'score':>6}  {'vrank':>5} {'krank':>5}  {'section':34} | source")
    for r in rows:
        vr = r["vrank"] if r["vrank"] is not None else "-"
        kr = r["krank"] if r["krank"] is not None else "-"
        print(f"  {r['score']:.4f}  {str(vr):>5} {str(kr):>5}  {r['section'][:34]:34} | {r['source']}")


if __name__ == "__main__":
    settings = get_settings()
    repo = DocumentRepository(embedder=Embedder(settings.openai_api_key.get_secret_value()))

    print("══ Part 1: scoping (vector-only) ══")
    print(f"QUERY: {QUERY!r}")
    _show_vec("(A) NO filter — expect mixed species (the problem):",
              repo.search(QUERY, k=6))
    _show_vec("(B) species='Humpback Whale' — expect humpback-only (the fix):",
              repo.search(QUERY, k=6, species="Humpback Whale"))

    # A PRECISE proper-noun term ("Rockwood", an author) — vectors under-weight a bare
    # token, the keyword channel pins it exactly, so hybrid reorders. (Note: keep the
    # keyword query short — websearch_to_tsquery ANDs bare terms, so a long NL sentence
    # rarely matches and hybrid silently collapses to vector-only.)
    hq = "Rockwood"
    print("\n\n══ Part 2: vector-only vs hybrid (same query, humpback-scoped) ══")
    print(f"QUERY: {hq!r}")
    _show_hybrid("(C) vector-only (vrank shows pure cosine order):",
                 [{**r, "vrank": i + 1, "krank": None, "score": 1.0 / (60 + i + 1)}
                  for i, r in enumerate(repo.search(hq, k=6, species="Humpback Whale"))])
    _show_hybrid("(D) hybrid RRF (vector + tsv): chunks strong in either channel rise:",
                 repo.search_hybrid(hq, k=6, species="Humpback Whale"))
