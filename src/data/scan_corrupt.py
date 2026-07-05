"""scan_corrupt.py — flag unreadable JPEGs in the manifest, ONCE.

A few of the downloaded images are truncated/unreadable: they downloaded fine
(status='ok') but PIL can't decode them, so they crash any code that opens them.
Rather than skip them at runtime everywhere, mark them ONCE here:

    status = 'ok'  →  status = 'corrupt'   (for images PIL can't open)

After this, read_pending (embed) and load_pool (train) exclude them automatically —
both already filter `status='ok'`, so no special-case code lives in the hot loops.
The trainer keeps a runtime skip as a backstop for newly downloaded bad files.

    uv run python -m data.scan_corrupt            # scan + mark
    uv run python -m data.scan_corrupt --dry-run  # report counts, change nothing
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from core.config import get_settings
from data.embed_images import resolve_path
from stores.postgres import get_pg_gateway

log = logging.getLogger(__name__)

THREADS = 8  # PIL decode releases the GIL, so threads give real parallel I/O


def is_readable(path: str) -> bool:
    """True if PIL can fully decode the file. verify() catches truncation/garbage."""
    try:
        with Image.open(path) as im:
            im.verify()  # checks the whole file, not just the header
        return True
    except Exception:  # noqa: BLE001 — any failure means "don't feed this to the model"
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="flag unreadable manifest images as status='corrupt'")
    ap.add_argument("--dry-run", action="store_true", help="report counts; make no changes")
    args = ap.parse_args()

    gw = get_pg_gateway()
    image_root = get_settings().image_root
    rows = gw.fetch_rows("SELECT sighting_id, asset_ref FROM manifest WHERE status = 'ok'")
    log.info("scanning %d images (status='ok')", len(rows))

    paths = [(r["sighting_id"], resolve_path(r["asset_ref"], image_root)) for r in rows]
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        ok_flags = list(ex.map(lambda sp: is_readable(sp[1]), paths))

    bad = [sid for (sid, _), ok in zip(paths, ok_flags) if not ok]
    log.info("found %d unreadable / %d scanned", len(bad), len(rows))

    if not bad:
        log.info("nothing to flag — every image is readable")
        return
    if args.dry_run:
        for sid, p in ((sid, p) for (sid, p), ok in zip(paths, ok_flags) if not ok):
            log.info("  would flag sighting %s → %s", sid, p)
        log.info("--dry-run: no changes written")
        return

    n = gw.execute(
        "UPDATE manifest SET status = 'corrupt' WHERE sighting_id = ANY(%s) AND status = 'ok'",
        (bad,),
    )
    log.info("marked %d rows status='corrupt'", n)


if __name__ == "__main__":
    main()
