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

import open_clip
import torch
from PIL import Image

log = logging.getLogger(__name__)

# --- the identity of this embedding run (stamped on every catalog row) ---
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
EMBEDDER_VER = "clip-vitb32-v1"  # the version tag; a query must match the catalog's
DIM = 512  # MUST equal the vector(DIM) column on fluke_embeddings


def pick_device() -> str:
    """Mac → MPS (Metal GPU); else CUDA; else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str):
    """Load the frozen model + its OWN preprocessing transform (load once, reuse)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()  # eval(): inference mode, no dropout/bn updates
    return model, preprocess


class ImageEmbedder:
    """Load CLIP once, embed many images into unit (cosine-space) vectors.

    Loading CLIP per image would be wasteful; construct one of these and reuse it
    (the build job holds one; the photo_id tool holds one).
    """

    def __init__(self, device: str | None = None):
        self.device = device or pick_device()
        self.model, self.preprocess = load_model(self.device)
        self.ver = EMBEDDER_VER
        log.info("ImageEmbedder ready (device=%s, ver=%s)", self.device, self.ver)

    def _encode(self, tensors: list) -> list[list[float]]:
        """One forward pass over a stacked batch → L2-normalized rows."""
        batch = torch.stack(tensors).to(self.device)  # [N, 3, H, W]
        with torch.no_grad():  # inference: no gradients → faster, less memory
            feats = self.model.encode_image(batch)  # [N, DIM]
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
