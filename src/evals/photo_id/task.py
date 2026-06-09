"""task.py — the system under test: photo_id retrieval on one held-out query.

Fetches the held-out sighting's STORED embedding (no image bytes, no CLIP load —
the catalog vectors already exist) and searches the live index with that vector
excluded, so the whale must be re-identified from its OTHER photos. Deterministic
and tokenless: Usage is always zero, and one sample per case suffices (repeats=1).
"""

from __future__ import annotations

from evals.framework.core import Usage
from tools.photo_id import PhotoIDResult, PhotoIDTool


class PhotoIDTask:
    """Task[PhotoIDInput, PhotoIDResult] — drive photo_id from a frozen split case."""

    def __init__(self, tool: PhotoIDTool | None = None, k: int = 10):
        # No embedder needed: we query by STORED vector. The tool loads CLIP lazily,
        # so this never pays the model-load cost.
        self.tool = tool or PhotoIDTool()
        # the experiment identity; LoRA reruns get a new ver → new label.
        # Taken from the tool so it reflects the actually-active version.
        self.label = f"photo_id-{self.tool.ver}"
        self.k = k  # return top-k individuals so Recall@1 and Recall@5 both read the same list

    def _fetch_vector(self, sighting_id: int):
        rows = self.tool.gw.fetch_rows(
            "SELECT embedding FROM fluke_embeddings WHERE sighting_id = %s AND embedder_ver = %s",
            (sighting_id, self.tool.ver),
        )
        return rows[0]["embedding"] if rows else None

    async def run(self, task_input: dict) -> tuple[PhotoIDResult | None, Usage]:
        sid = task_input["query_sighting_id"]
        qvec = self._fetch_vector(sid)
        if qvec is None:
            return None, Usage()  # vector missing → a scored parse-failure, not a crash
        result = self.tool.query_by_vector(qvec, k=self.k, exclude_sighting_id=sid)
        return result, Usage(total_tokens=0)
