"""Shared fixtures for the test suite.

The live integration tests hit the live Postgres DB + on-disk image catalog (no Anthropic
calls → no token cost). They self-skip when the data isn't reachable, so the suite is safe
to run anywhere. Run only the live suite:  uv run pytest -m live
"""

from __future__ import annotations

import pytest

# A known, stable reference animal: #479, a Monterey Bay humpback (192 sightings) whose
# core range sits inside the Monterey Bay NMS — the F5 ship-strike reference case.
KNOWN_INDIVIDUAL_ID = 479
# Its core-range bbox, so vessel_traffic can be exercised without first running sighting_lookup.
MONTEREY_BBOX = {"lat_min": 36.59, "lat_max": 36.86, "lon_min": -122.04, "lon_max": -121.84}


@pytest.fixture
def known_individual_id() -> int:
    return KNOWN_INDIVIDUAL_ID


@pytest.fixture
def monterey_bbox() -> dict:
    return dict(MONTEREY_BBOX)


@pytest.fixture
def catalog_image() -> str:
    """Path to one of #479's catalog photos; skip if the image catalog isn't present."""
    from config import get_settings

    d = get_settings().image_root / str(KNOWN_INDIVIDUAL_ID)
    if not d.is_dir():
        pytest.skip(f"no image catalog at {d}")
    jpgs = sorted(d.glob("*.jpg"))
    if not jpgs:
        pytest.skip(f"no .jpg images under {d}")
    return str(jpgs[0])


@pytest.fixture
def image_url():
    """Serve the local image catalog over http and yield a URL to a #479 photo.

    Exercises the embedder's URL-fetch path deterministically — no dependency on an
    external host staying up.
    """
    import functools
    import http.server
    import socketserver
    import threading

    from config import get_settings

    root = get_settings().image_root
    d = root / str(KNOWN_INDIVIDUAL_ID)
    jpgs = sorted(d.glob("*.jpg")) if d.is_dir() else []
    if not jpgs:
        pytest.skip(f"no .jpg images under {d}")

    class _Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request stderr logging
            pass

    handler = functools.partial(_Quiet, directory=str(root))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)  # port 0 → free port
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}/{KNOWN_INDIVIDUAL_ID}/{jpgs[0].name}"
    finally:
        httpd.shutdown()
