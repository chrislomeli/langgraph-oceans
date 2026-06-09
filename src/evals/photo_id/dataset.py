"""dataset.py — the held-out cases, read from the FROZEN split table.

`load()` does not select or sample anything stochastic — it just projects the
already-frozen `datasets.photo_id_eval_split` (built once by build_split.sql) into
framework `Case`s. That separation is the whole point: the split is the fixed
yardstick; this is a read of it. Re-seeding LangSmith from here always yields the
same cases as long as the table is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from evals.framework.core import Case
from stores.postgres import get_pg_gateway

# Input/expected travel as plain dicts (LangSmith stores JSON; input_model=None).
PhotoIDInput = dict   # {"query_sighting_id": int, "individual_id": int, "name": str | None}
PhotoIDExpected = dict  # {"individual_id": int, "name": str | None}


@dataclass
class PhotoIDLeaveOneOut:
    """DatasetSource: one closed-set leave-one-out case per held-out individual.

    A case = "here is one sighting of a known whale; hide it, do its OTHER photos
    still rank the whale?" The truth is the individual_id; the Task supplies the
    ranked candidates; RetrievalRanking scores Recall@k / MRR.
    """

    name: str = "photo-id-loo-closed-set"
    # The embedder version is part of the dataset identity: a LoRA re-embed (…-v2)
    # produces a DIFFERENT catalog, so it's a new baseline against the same split.
    version: str = "v1-clip-vitb32"
    role: str = "closed_set"
    limit: int = 0  # 0 = all; >0 = a small deterministic subset for a smoke test

    def __post_init__(self):
        # A smoke subset is its OWN dataset (distinct name) so it can never pollute
        # or alias the full baseline dataset in LangSmith.
        if self.limit:
            self.name = f"{self.name}-smoke{self.limit}"

    def load(self) -> list[Case[PhotoIDInput, PhotoIDExpected]]:
        gw = get_pg_gateway()
        rows = gw.fetch_rows(
            """
            SELECT sp.individual_id, sp.query_sighting_id, i.name
            FROM photo_id_eval_split sp
            JOIN individuals i USING (individual_id)
            WHERE sp.role = %s
            ORDER BY sp.individual_id
            """,
            (self.role,),
        )
        if self.limit:
            rows = rows[: self.limit]
        return [
            Case(
                id=f"loo-{r['query_sighting_id']}",
                input={
                    "query_sighting_id": r["query_sighting_id"],
                    "individual_id": r["individual_id"],
                    "name": r["name"],
                },
                expected={"individual_id": r["individual_id"], "name": r["name"]},
                tags=(self.role,),
            )
            for r in rows
        ]
