"""embed_images.py — turn downloaded JPEGs into vectors in fluke_embeddings.

  Step 3 of the pipeline (download → EMBED → query).
  Idempotent per embedder_ver: re-running skips sightings already embedded
  at THIS version; switching the version re-embeds without wiping the old.
  """

import itertools
import logging
from pathlib import Path

from pgvector import Vector

from config import get_settings
from models.image_embedder import EMBEDDER_VER, ImageEmbedder
from stores.postgres import get_pg_gateway

log = logging.getLogger(__name__)

# manifest.asset_ref carries a legacy "blobs/" prefix (e.g. "blobs/4606/6134.jpg")
# from when images lived under src/data/. The catalog now lives outside the repo
# at Settings.image_root, laid out <individual_id>/<sighting_id>.jpg — so we drop
# that prefix when resolving a row to an on-disk path.
def resolve_path(asset_ref: str, image_root: Path) -> str:
    rel = Path(asset_ref)
    if rel.parts and rel.parts[0] == "blobs":
        rel = Path(*rel.parts[1:])
    return str(image_root / rel)

# The model identity (MODEL_NAME / EMBEDDER_VER / DIM) and the embed math now live
# in the Layer-5 ImageEmbedder, so this build job and the photo_id tool embed
# IDENTICALLY. This script's job is just plumbing: pick pending rows → embed → write.
BATCH_SIZE = 32
LIMIT = None   # smoke test on N images; set to None for the full run


def read_pending(gw):
    """Generator: sightings that still need a vector at THIS embedder_ver.

    Joins manifest -> sightings for the license filter, and excludes any
    sighting already embedded at this version (idempotency).
    """
    sql = """
          SELECT m.sighting_id, m.individual_id, m.asset_ref, m.source_url
          FROM manifest m
                   JOIN sightings s ON s.sighting_id = m.sighting_id
          WHERE s.license NOT LIKE '%%-nd/%%' -- ML-safety: drop no-derivatives
            AND m.status = 'ok'               -- only successfully downloaded
            AND NOT EXISTS ( -- skip already-embedded-at-this-version
              SELECT 1 \
              FROM fluke_embeddings fe
              WHERE fe.sighting_id = m.sighting_id
                AND fe.embedder_ver = %s) \
          """
    # TODO: stream these (server-side cursor) if the result set is large;
    #       for ~110k small rows, fetch_rows is fine.
    yield from gw.fetch_rows(sql, (EMBEDDER_VER,))


def batched(iterable, n):
    """Yield lists of up to n items — turns the row stream into batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_batch(rows, embedder: ImageEmbedder, image_root: Path) -> list:
    """Embed a batch of manifest rows → (row, vector) pairs, identity preserved.

    Delegates the model work to the shared Layer-5 embedder; here we just bridge
    rows ↔ blob paths. Unreadable files are skipped inside embed_paths.
    """
    path_to_row = {resolve_path(r["asset_ref"], image_root): r for r in rows}
    pairs = embedder.embed_paths(list(path_to_row))
    return [(path_to_row[p], vec) for p, vec in pairs]


def write_batch(gw, pairs) -> None:
    """Insert (identity + vector) rows into fluke_embeddings."""
    if not pairs:
        return
    sql = """
          INSERT INTO fluke_embeddings
          (individual_id, sighting_id, asset_ref, source_url, embedding, embedder_ver)
          VALUES (%s, %s, %s, %s, %s, %s)
          ON CONFLICT (sighting_id, embedder_ver) DO NOTHING \
          """
    params = [
        (
            row["individual_id"],
            row["sighting_id"],
            row["asset_ref"],
            row["source_url"],
            Vector(vec),  # wrap the plain list so pgvector stores it as a vector
            EMBEDDER_VER,
        )
        for row, vec in pairs
    ]
    with gw.conn() as conn, conn.cursor() as cur:
        cur.executemany(sql, params)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    settings.apply_hf()  # export HF_TOKEN so Hub downloads are authenticated
    image_root = settings.image_root
    embedder = ImageEmbedder()
    gw = get_pg_gateway()

    source = read_pending(gw)
    if LIMIT:
        source = itertools.islice(source, LIMIT)

    done = 0
    for rows in batched(source, BATCH_SIZE):
        pairs = embed_batch(rows, embedder, image_root)
        write_batch(gw, pairs)
        done += len(pairs)
        log.info("embedded %d", done)

if __name__ == "__main__":
    main()
