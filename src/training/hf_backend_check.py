"""hf_backend_check.py — throwaway checkpoint for the "HF backend" tutorial box.

Proves the new transformers `hf` backend in ImageEmbedder works AND lines up with the
existing open_clip path. Delete once the box is checked.

What it asserts (the checkpoint):
  1. clip-hf-vitb32-v1 loads and embeds without error  → the hf path is wired
  2. its vectors are 512-d and unit-norm               → same contract as open_clip
  3. each hf vector is HIGHLY similar (cosine) to the   → same weights, same space
     open_clip vector for the SAME image

Note on #3: we expect HIGH but NOT 1.0. open_clip and transformers load the same OpenAI
weights but preprocess slightly differently (resize/normalize details). That residual
gap is EXACTLY why we register a separate `clip-hf-vitb32-v1` baseline instead of reusing
the open_clip number — so the LoRA lift is measured on one consistent loader.

    uv run python -m training.hf_backend_check
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import get_settings
from models.image_embedder import ImageEmbedder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

N_IMAGES = 6           # a handful is plenty to prove the path
COSINE_FLOOR = 0.90    # hf-vs-open_clip similarity we expect to clear (high, not 1.0)


def _sample_images(n: int) -> list[Path]:
    """A few real catalog JPEGs from Settings.image_root (<individual>/<sighting>.jpg)."""
    root = get_settings().image_root
    paths = sorted(root.glob("*/*.jpg"))[:n]   # sorted → deterministic pick
    if not paths:
        raise SystemExit(
            f"No images under {root} — they may be on S3 only. "
            "Pull a few locally (or point IMAGE_ROOT at a local copy) to run this check."
        )
    return paths


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product = cosine, since ImageEmbedder returns unit-norm vectors."""
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    get_settings().apply_hf()   # export HF token if set (openai/clip is public, but safe)
    paths = _sample_images(N_IMAGES)
    log.info("checking on %d images from %s", len(paths), paths[0].parent.parent)

    # Embed the SAME images with both backends. embed_paths returns (path, vector),
    # only for loadable files, so key by path to align the two runs safely.
    oc = dict(ImageEmbedder("clip-vitb32-v1").embed_paths(paths))     # open_clip baseline
    hf = dict(ImageEmbedder("clip-hf-vitb32-v1").embed_paths(paths))  # NEW hf backend

    shared = [p for p in paths if p in oc and p in hf]
    dims_ok = all(len(hf[p]) == 512 for p in shared)
    norms = [_cosine(hf[p], hf[p]) ** 0.5 for p in shared]            # ||v|| = sqrt(v·v)
    norm_ok = all(abs(nrm - 1.0) < 1e-3 for nrm in norms)
    cosines = [_cosine(oc[p], hf[p]) for p in shared]
    mean_cos = sum(cosines) / len(cosines) if cosines else 0.0
    cos_ok = mean_cos >= COSINE_FLOOR

    print("\n===== HF backend check =====")
    print(f"  images embedded both ways : {len(shared)}/{len(paths)}")
    print(f"  hf vectors are 512-d      : {dims_ok}")
    print(f"  hf vectors are unit-norm  : {norm_ok}  (norms {min(norms):.4f}–{max(norms):.4f})")
    print(f"  cosine(open_clip, hf)     : mean={mean_cos:.4f}  "
          f"(range {min(cosines):.4f}–{max(cosines):.4f}, floor {COSINE_FLOOR})")

    if dims_ok and norm_ok and cos_ok:
        print("\n  ✅ HF backend checkpoint PASSED — clip-hf-vitb32-v1 is wired and consistent")
        print("     (cosine < 1.0 is expected & is why we keep a separate hf baseline)")
    else:
        print("\n  ✗ checkpoint FAILED — see which line above is False")
        if not cos_ok:
            print("     low cosine usually = preprocessing mismatch (the silent-degradation trap)")


if __name__ == "__main__":
    main()
