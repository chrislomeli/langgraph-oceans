"""photo_id.py — Layer 3 perception tool: "which catalogued individual is this whale photo?"

The agent's eyes. Roadmap item 5. Flow:

  query image
    → embed with the SAME model as the catalog (shared space; via Layer-5 ImageEmbedder)
    → cosine nearest-neighbor over fluke_embeddings via the HNSW index   (top-N images)
    → aggregate those images to ranked individuals (best image per whale)
    → abstain ("NOVEL") if nothing is confidently close

Dumb-tool contract (see contracts.py): this tool SURFACES signals — ranked
candidates, `margin` (how clearly #1 beats #2), and `abstain` (top score below
threshold) — but it does NOT decide what to do about them. The agent (Layer 2)
owns that judgment: disambiguate on a small margin, route to recovery on abstain.
On an empty catalog it returns `ok=False`, never raises.

No image bytes required for the catalog: the 114K catalog vectors already live in
`fluke_embeddings`, so search is pure vector math. `query_by_vector` is the core
(used by Layer-A eval's leave-one-out, which reuses STORED vectors — no JPEGs);
`query_by_image` is the thin live-query wrapper that embeds one new photo. Catalog
image bytes are only needed again for Phase C (LoRA). Citations point at the live
public `thumb_url`, not local files.
"""

import logging

from pgvector import Vector
from pydantic import BaseModel, Field

from config import get_settings
from models.image_embedder import ImageEmbedder
from stores.postgres import get_pg_gateway
from tools.contracts import Citation, Filters, ToolResult

log = logging.getLogger(__name__)

# How many nearest IMAGES to pull before collapsing to individuals. Bigger = less
# chance of missing a whale whose best photo isn't in the very top; 50 is ample
# for a top-5 answer.
N_CANDIDATE_IMAGES = 50

# Below this cosine similarity, declare NOVEL instead of guessing an identity.
# PLACEHOLDER — calibrated for real in the Layer-A eval (roadmap item 6). Raw
# cosine isn't a probability; a real threshold comes from held-out data.
ABSTAIN_THRESHOLD = 0.80

CATALOG = "happywhale-via-obis"

# `individuals` is bigint-keyed and carries `name` (the hub migration landed
# 2026-06-08), so fe.individual_id joins it directly and Candidate.name lights up.
INDIVIDUALS_HAS_BIGINT_KEY = True


class Candidate(BaseModel):
    """One ranked individual the photo might be."""

    individual_id: int  # the bigint hub key; what sighting_lookup will take
    name: str | None = None  # organism_name via the individuals hub (once it's bigint-keyed)
    score: float  # best cosine similarity to the query (calibrated later)
    catalog: str = CATALOG
    thumb_ref: str | None = None  # public thumb_url of the matched catalog image (multimodal citation)
    n_images: int = 1  # support: how many of the top-N images were this whale


class PhotoIDResult(ToolResult):
    """Ranked candidates + the two judgment signals the agent reasons over."""

    candidates: list[Candidate] = Field(default_factory=list)  # ranked, best first
    abstain: bool = True  # top score < threshold → treat as NOVEL  (open→closed seam)
    margin: float | None = None  # score(top1) − score(top2): disambiguation signal
    threshold: float = ABSTAIN_THRESHOLD  # echoed for eval reproducibility


def _search_nearest_images(gw, qvec, ver, n=N_CANDIDATE_IMAGES, exclude_sighting_id=None) -> list[dict]:
    """Top-N nearest catalog IMAGES by cosine — uses the HNSW index.

    Joins sightings for the public thumb_url (the multimodal citation target) and,
    once individuals is bigint-keyed, the individual's name. exclude_sighting_id
    drops that one image (leave-one-out), so a query can't match itself in eval.
    """
    qv = Vector(qvec)
    # The f-strings inject only FIXED clause strings (no user data); ids still
    # travel as %s parameters, so this is not SQL injection.
    exclude_clause = "AND fe.sighting_id <> %s" if exclude_sighting_id is not None else ""
    sql = f"""
        SELECT fe.individual_id,
               fe.sighting_id,
               s.thumb_url,
               i.name AS individual_name,
               1 - (fe.embedding <=> %s) AS score   -- cosine similarity (1 - cosine distance)
        FROM fluke_embeddings fe
        JOIN sightings s ON s.sighting_id = fe.sighting_id
        LEFT JOIN individuals i ON i.individual_id = fe.individual_id
        WHERE fe.embedder_ver = %s
          {exclude_clause}
        ORDER BY fe.embedding <=> %s              -- <=> is cosine distance; HNSW accelerates this
        LIMIT %s
    """
    params = [qv, ver]
    if exclude_sighting_id is not None:
        params.append(exclude_sighting_id)
    params.extend([qv, n])
    return gw.fetch_rows(sql, tuple(params))


def _aggregate_to_individuals(rows, k=5) -> list[Candidate]:
    """Collapse nearest images → ranked individuals, keeping each whale's BEST image."""
    best: dict[int, Candidate] = {}
    for r in rows:
        ind = r["individual_id"]
        if ind not in best:
            best[ind] = Candidate(
                individual_id=ind,
                name=r.get("individual_name"),
                score=r["score"],
                thumb_ref=r.get("thumb_url"),
                n_images=1,
            )
        else:
            c = best[ind]
            c.n_images += 1
            if r["score"] > c.score:  # a whale's score = its closest photo
                c.score = r["score"]
                c.thumb_ref = r.get("thumb_url")
    ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
    return ranked[:k]


class PhotoIDTool:
    """The photo_id tool: open the pool ONCE, answer many queries.

    Two entry points: `query_by_vector` (the core — used by eval's leave-one-out
    on STORED vectors, no image bytes) and `query_by_image` (live use — embeds one
    new photo, then delegates). The embedder is loaded lazily, so vector-only
    workloads (eval) never pay the CLIP load cost.
    """

    def __init__(self, ver: str | None = None, embedder: ImageEmbedder | None = None):
        # ver defaults from Settings (env-overridable) so A/B needs no code edit;
        # if an embedder is supplied, defer to ITS version so the two can't disagree.
        self.ver = embedder.ver if embedder is not None else (ver or get_settings().embedder_ver)
        self._embedder = embedder  # lazy: only needed by query_by_image
        self.gw = get_pg_gateway()
        log.info("PhotoIDTool ready (ver=%s)", self.ver)

    @property
    def embedder(self) -> ImageEmbedder:
        if self._embedder is None:
            self._embedder = ImageEmbedder(ver=self.ver)
        return self._embedder

    def query_by_vector(
        self,
        qvec,
        k: int = 5,
        filters: Filters | None = None,  # noqa: ARG002 — accepted per the shared contract; see note
        exclude_sighting_id: int | None = None,
    ) -> PhotoIDResult:
        # NOTE: `filters` is accepted to honor the one-contract rule but not yet
        # applied — species/region/date would filter by JOINing fluke_embeddings →
        # sightings → individuals. Deferred to the agent-integration step (the
        # closed-set demo doesn't need it yet); the seam is here so it's free later.
        rows = _search_nearest_images(self.gw, qvec, self.ver, exclude_sighting_id=exclude_sighting_id)
        ranked = _aggregate_to_individuals(rows, k)

        top_score = ranked[0].score if ranked else None
        margin = (ranked[0].score - ranked[1].score) if len(ranked) >= 2 else None
        abstain = (top_score is None) or (top_score < ABSTAIN_THRESHOLD)

        citations = [
            Citation(
                kind="catalog_image",
                source=c.catalog,
                locator=c.thumb_ref,
                ref=str(c.individual_id),
            )
            for c in ranked
        ]
        if top_score is None:
            summary = "Empty catalog / no neighbors found"
        elif abstain:
            summary = f"No confident match (top={top_score:.3f} < {ABSTAIN_THRESHOLD}) → likely NOVEL"
        else:
            summary = f"Top match individual {ranked[0].individual_id} (score={top_score:.3f}, margin={margin})"

        return PhotoIDResult(
            tool="photo_id",
            ok=bool(ranked),
            summary=summary,
            citations=citations,
            candidates=ranked,
            abstain=abstain,
            margin=margin,
            threshold=ABSTAIN_THRESHOLD,
        )

    def query_by_image(
        self,
        image_path,
        k: int = 5,
        filters: Filters | None = None,
        exclude_sighting_id: int | None = None,
    ) -> PhotoIDResult:
        """Live query: embed one new photo (or a fetched URL), then search."""
        qvec = self.embedder.embed_one(image_path)
        return self.query_by_vector(qvec, k=k, filters=filters, exclude_sighting_id=exclude_sighting_id)


def _print_result(result: PhotoIDResult) -> None:
    print(f"  ok={result.ok}  abstain(NOVEL)? {result.abstain}  margin={result.margin}")
    print(f"  summary: {result.summary}")
    for i, c in enumerate(result.candidates, 1):
        nm = f" {c.name!r}" if c.name else ""
        print(
            f"    {i}. individual {c.individual_id:>8}{nm}  score={c.score:.4f}  "
            f"({c.n_images} of top-{N_CANDIDATE_IMAGES} imgs)"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Leave-one-out demo — entirely from STORED vectors (no blobs, no CLIP load):
    # grab one catalog embedding of a multi-photo whale, use it as the query, then
    # HIDE its own sighting and check the whale is still re-identified from its
    # OTHER photos. This is exactly the Layer-A eval move (item 6).
    tool = PhotoIDTool()
    gw = tool.gw
    pick = gw.fetch_rows(
        """
        SELECT fe.individual_id, fe.sighting_id, fe.embedding
        FROM fluke_embeddings fe
        WHERE fe.embedder_ver = %s
          AND fe.individual_id IN (
            SELECT individual_id FROM fluke_embeddings
            WHERE embedder_ver = %s GROUP BY individual_id HAVING count(*) >= 6 LIMIT 1)
        LIMIT 1
        """,
        (tool.ver, tool.ver),
    )[0]
    true_individual, query_sighting, qvec = pick["individual_id"], pick["sighting_id"], pick["embedding"]
    print(f"\nquery vector: stored embedding of individual {true_individual}, sighting {query_sighting}")

    print("\n--- normal search (the query's own vector IS in the catalog → trivial self-match) ---")
    _print_result(tool.query_by_vector(qvec))

    print(f"\n--- leave-one-out (exclude sighting {query_sighting} → re-ID from OTHER photos) ---")
    loo = tool.query_by_vector(qvec, exclude_sighting_id=query_sighting)
    _print_result(loo)

    rank = next((i for i, c in enumerate(loo.candidates, 1) if c.individual_id == true_individual), None)
    if rank:
        print(f"\n  ✓ RE-IDENTIFIED: individual {true_individual} still ranked #{rank} without its own photo")
    else:
        print(f"\n  ✗ MISSED: individual {true_individual} not in top {len(loo.candidates)} once its photo was hidden")
