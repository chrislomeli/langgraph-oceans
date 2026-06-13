"""reid_dataset.py — the by-individual generalization cases, read from the FROZEN split.

Projects `datasets.reid_split` (built once by build_reid_split.sql) into framework
`Case`s, exactly as dataset.py projects `photo_id_eval_split`. One case per val/test
whale: its deterministic held-out `query_sighting_id` is the query; its OTHER photos
are the gallery it must be re-found in. Nothing stochastic — re-seeding reproduces
the same cases while the table is unchanged.

The split is keyed on individual/sighting ids, so it is EMBEDDER-INDEPENDENT: the
baseline run and the LoRA run score the same whales, only the vectors differ
(`Settings.embedder_ver` is the A/B axis).
"""

from __future__ import annotations

from dataclasses import dataclass

from evals.framework.core import Case
from stores.postgres import get_pg_gateway

ReIDInput = dict   # {"query_sighting_id": int, "individual_id": int, "name": str | None}
ReIDExpected = dict  # {"individual_id": int, "name": str | None}


@dataclass
class ReIDSplitDataset:
    """DatasetSource over datasets.reid_split — one case per val/test query whale.

    `split='val'` while iterating on LoRA; `split='test'` exactly once at the end
    (eval.py enforces the discipline with --final). The gallery for a query is every
    individual in the SAME split — the disjoint protocol — supplied to the Task via
    `gallery_individual_ids()`, never decided by the tool.
    """

    split: str = "val"
    name: str = "photo-id-reid-byindividual"
    version: str = "v1"  # versions the SPLIT, not the embedder (that's the Task's ver)
    limit: int = 0  # 0 = all; >0 = a small deterministic subset for a smoke test

    def __post_init__(self):
        assert self.split in ("val", "test"), f"split must be val|test, got {self.split!r}"
        self.name = f"{self.name}-{self.split}"
        # A smoke subset is its OWN dataset (distinct name) so it can never pollute
        # or alias the full dataset in LangSmith.
        if self.limit:
            self.name = f"{self.name}-smoke{self.limit}"

    def gallery_individual_ids(self) -> list[int]:
        """The candidate pool: every individual in THIS split (the disjoint gallery).

        Includes whales beyond the smoke `limit` on purpose — the gallery is the
        split, the limit only caps how many queries we score.
        """
        rows = get_pg_gateway().fetch_rows(
            "SELECT individual_id FROM datasets.reid_split WHERE split = %s ORDER BY individual_id",
            (self.split,),
        )
        return [r["individual_id"] for r in rows]

    def load(self) -> list[Case[ReIDInput, ReIDExpected]]:
        rows = get_pg_gateway().fetch_rows(
            """
            SELECT sp.individual_id, sp.query_sighting_id, i.name
            FROM datasets.reid_split sp
            JOIN individuals i USING (individual_id)
            WHERE sp.split = %s
              AND sp.query_sighting_id IS NOT NULL
            ORDER BY sp.individual_id
            """,
            (self.split,),
        )
        if self.limit:
            rows = rows[: self.limit]
        return [
            Case(
                id=f"reid-{self.split}-{r['query_sighting_id']}",
                input={
                    "query_sighting_id": r["query_sighting_id"],
                    "individual_id": r["individual_id"],
                    "name": r["name"],
                },
                expected={"individual_id": r["individual_id"], "name": r["name"]},
                tags=(f"reid_{self.split}",),
            )
            for r in rows
        ]
