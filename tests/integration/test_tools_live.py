"""Live per-tool integration tests — each agents tool, called for real against the DB.

One test per bound tool (`agents/tools.py`). Each exercises the FULL path: the `@tool`
wrapper → the underlying Layer-3 tool → live Postgres (+ the image embedder for photo_id).
No LLM is involved, so these cost nothing and run without an API key. They'd have caught
the `embedder_ver` config regression that broke photo_id.

Run:  uv run pytest -m live
"""

import pytest

from agents.tools import photo_id, sighting_context, sighting_lookup, vessel_traffic

pytestmark = pytest.mark.live


def test_photo_id_identifies_known_individual(catalog_image):
    out = photo_id.invoke({"image_path": catalog_image})
    assert isinstance(out, str) and out
    # self-query of #479's own photo → #479 should be the top match
    assert "479" in out


def test_photo_id_accepts_url(image_url):
    # same #479 photo, but served over http → exercises the embedder's URL-fetch path
    out = photo_id.invoke({"image_path": image_url})
    assert isinstance(out, str) and out
    assert "479" in out


def test_sighting_lookup_returns_history_and_range(known_individual_id):
    out = sighting_lookup.invoke({"individual_id": known_individual_id})
    assert "Humpback" in out          # #479 is a humpback
    assert "range_bbox" in out        # the bbox seam that feeds vessel_traffic


def test_sighting_context_returns_habitat(known_individual_id):
    out = sighting_context.invoke({"individual_id": known_individual_id})
    assert "sighting" in out.lower()
    assert "depth" in out.lower()


def test_vessel_traffic_reads_traffic_over_bbox(monterey_bbox):
    out = vessel_traffic.invoke({**monterey_bbox, "year": 2024})
    assert "transit" in out.lower()
    assert "Monterey" in out          # the bbox overlaps Monterey Bay NMS
