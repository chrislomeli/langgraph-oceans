"""image_embedder.py — Layer 5: the CLIP image embedder (one source of truth).

Both the catalog build (`data/embed_images.py`) and the `photo_id` tool
(`tools/photo_id.py`) embed images by importing from HERE, so a query vector and
the catalog vectors are always made by the same model and share one space + one
`EMBEDDER_VER`. A query vector can only be compared to catalog vectors made by
the same model — funnelling both through this module makes a version mismatch
structurally impossible.

We are NOT training the model — this is pure inference (image → 512 numbers).
"""

import logging
from dataclasses import dataclass

import open_clip
import torch
from PIL import Image

log = logging.getLogger(__name__)


# --- the embedder registry: version tag → how to build that model ---------------
# A version tag is BOTH the value stamped on every catalog row AND the key that
# selects which model/weights produce those vectors — so a row's vector and the
# model that made it can never drift apart. The ACTIVE tag lives in Settings
# (embedder_ver, override via the EMBEDDER_VER env var); this module just maps a
# tag → build spec. To A/B a new model: add an entry here, point Settings at it,
# and re-run embed_images.py (it re-embeds alongside the old version).
@dataclass(frozen=True)
class EmbedderSpec:
    model_name: str          # open_clip architecture ("ViT-B-32") or "hf-hub:<repo>"
    pretrained: str | None   # open_clip pretrained tag; None for hf-hub checkpoints
    dim: int                 # embedding width; MUST equal the vector(DIM) column


EMBEDDERS: dict[str, EmbedderSpec] = {
    "clip-vitb32-v1": EmbedderSpec("ViT-B-32", "openai", 512),
    # bioCLIP — native open_clip ViT-B/16 trained on TreeOfLife-10M; still 512-d.
    "bioclip-v1": EmbedderSpec("hf-hub:imageomics/bioclip", None, 512),
}


def resolve_spec(ver: str) -> EmbedderSpec:
    """Map a version tag → its build spec, with a clear error for unknown tags."""
    try:
        return EMBEDDERS[ver]
    except KeyError:
        raise KeyError(f"unknown embedder_ver {ver!r}; known: {sorted(EMBEDDERS)}") from None


def pick_device() -> str:
    """Mac → MPS (Metal GPU); else CUDA; else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(spec: EmbedderSpec, device: str):
    """Load the frozen model + its OWN preprocessing transform (load once, reuse).

    The architecture/weights come from `spec` (chosen by the caller's version tag),
    so the model that runs here is always the one that tag names.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.model_name, pretrained=spec.pretrained
    )
    model = model.to(device).eval()  # eval(): inference mode, no dropout/bn updates
    return model, preprocess


class ImageEmbedder:
    """Load CLIP once, embed many images into unit (cosine-space) vectors.

    Loading CLIP per image would be wasteful; construct one of these and reuse it
    (the build job holds one; the photo_id tool holds one).
    """

    def __init__(self, ver: str, device: str | None = None):
        self.ver = ver                       # the identity stamped on every vector
        self.spec = resolve_spec(ver)        # which model/weights this tag means
        self.dim = self.spec.dim             # expected embedding width
        self.device = device or pick_device()
        self.model, self.preprocess = load_model(self.spec, self.device)
        log.info("ImageEmbedder ready (device=%s, ver=%s)", self.device, self.ver)

    def _encode(self, tensors: list) -> list[list[float]]:
        """One forward pass over a stacked batch → L2-normalized rows."""
        batch = torch.stack(tensors).to(self.device)  # [N, 3, H, W]
        with torch.no_grad():  # inference: no gradients → faster, less memory
            feats = self.model.encode_image(batch)  # [N, dim]
            # Guard: the model's real output width must match the version's declared
            # dim (and the vector(dim) column). Catches a mis-specified registry
            # entry here with a clear error, not as an opaque insert failure later.
            if feats.shape[-1] != self.dim:
                raise ValueError(
                    f"{self.ver!r} produced {feats.shape[-1]}-d embeddings "
                    f"but expects {self.dim} (the fluke_embeddings vector column)"
                )
            feats = feats / feats.norm(dim=-1, keepdim=True)  # unit length → cosine space
        return feats.cpu().tolist()

    def embed_paths(self, paths: list) -> list[tuple]:
        """Embed a batch of image paths → list of (path, unit-vector).

        Returns only the *loadable* paths (unreadable files are warned + skipped),
        so the caller must zip on the returned path, not assume 1:1 with input.
        """
        tensors, kept = [], []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(self.preprocess(img))  # model's own transform → tensor
                kept.append(p)
            except Exception as e:
                log.warning("unreadable %s: %s", p, e)
        if not tensors:
            return []
        return list(zip(kept, self._encode(tensors)))

    def embed_one(self, path) -> list[float]:
        """Embed a single image → one unit vector of length DIM (raises if unreadable)."""
        img = Image.open(path).convert("RGB")
        return self._encode([self.preprocess(img)])[0]
