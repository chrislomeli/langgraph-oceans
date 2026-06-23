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
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import open_clip
import torch
from PIL import Image

log = logging.getLogger(__name__)


def _open_rgb(src) -> Image.Image:
    """Open an image as RGB from a local path OR an http(s) URL.

    The catalog build passes local paths; a photo_id live query may pass a public URL
    (e.g. a Happywhale thumb_url), so the query path transparently fetches it. Keeping
    this in the embedder (not a separate agent tool) means fetching is below-the-agent
    plumbing, not a reasoning hop.
    """
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        with urlopen(src, timeout=15) as resp:  # noqa: S310 — trusted image URLs
            return Image.open(BytesIO(resp.read())).convert("RGB")
    return Image.open(src).convert("RGB")


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
    backend: str = "open_clip"        # "open_clip" | "hf" — which loader builds the model
    adapter_path: str | None = None   # peft (LoRA) checkpoint dir; only for hf LoRA vers


EMBEDDERS: dict[str, EmbedderSpec] = {
    "clip-vitb32-v1": EmbedderSpec(
        "ViT-B-32",
        "openai",
        512),
    # bioCLIP — native open_clip ViT-B/16 trained on TreeOfLife-10M; still 512-d.
    "bioclip-v1": EmbedderSpec("hf-hub:imageomics/bioclip", None, 512),
    # HuggingFace CLIP (same OpenAI ViT-B/32 weights, transformers loader). This is the
    # FAIR baseline for the LoRA lift: LoRA trains on the hf backend (peft can target its
    # q/k/v/out_proj), so the honest "before" must use the same loader, adapters off.
    "clip-hf-vitb32-v1": EmbedderSpec(
        "openai/clip-vit-base-patch32",
        None,
        512,
        backend="hf"),
    # The trained LoRA version: same hf base + an adapter checkpoint. Registered now so the
    # seam exists; the adapter dir is produced later (Part 4 / lora_train.py).
    # v3: off CLIP — EfficientNetV2-S @384 + ArcFace, 512-d embedding (reuses vector(512)).
    # The resolution probe showed resolution is the lever; this backbone takes 384px and
    # trains with the re-ID standard loss. adapter_path is set once train_arcface.py saves.
    "effnetv2s-arcface-v3": EmbedderSpec(
        "tf_efficientnetv2_s.in21k_ft_in1k",
        None,
        512,
        backend="timm",
        adapter_path="artifacts/reid/effnetv2s-arcface-v3.pt",  # trained state_dict
    ),
    # Part-4 scaled run (all train whales, P16K4, 3000 steps, best-probe checkpoint).
    "clip-hf-vitb32-lora-v2": EmbedderSpec(
        "openai/clip-vit-base-patch32",
        None,
        512,
        backend="hf",
        adapter_path="artifacts/lora/clip-hf-vitb32-lora-v2",
    ),
    "clip-hf-vitb32-lora-v1": EmbedderSpec(
        "openai/clip-vit-base-patch32",
        None,
        512,
        backend="hf",
        adapter_path="artifacts/lora/clip-hf-vitb32-lora-v1",
    ),
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
    so the model that runs here is always the one that tag names. Dispatches on
    `spec.backend`; BOTH branches return the same contract — a model `_encode` knows
    how to call, and a `preprocess(pil) -> [3,H,W] tensor` callable.
    """
    if spec.backend == "hf":
        return _load_hf(spec, device)
    if spec.backend == "timm":
        return _load_timm(spec, device)
    # open_clip path (today): the original baseline + bioCLIP.
    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.model_name, pretrained=spec.pretrained
    )
    model = model.to(device).eval()  # eval(): inference mode, no dropout/bn updates
    return model, preprocess


def _load_timm(spec: EmbedderSpec, device: str):
    """EfficientNetV2-S re-ID model (+ optional trained weights) — the `timm` backend.

    Same (model, preprocess) contract as the other backends. `adapter_path` here points
    at a saved state_dict (.pt) of trained weights; when absent the backbone is ImageNet-
    pretrained with a random head (plumbing-smoke only, not a useful embedder yet).
    """
    from models.reid_model import ReIDModel, build_preprocess

    model = ReIDModel(backbone=spec.model_name, emb_dim=spec.dim, pretrained=True).to(device).eval()
    if spec.adapter_path:
        weights = Path(spec.adapter_path)
        if not weights.exists():
            weights = Path(__file__).resolve().parents[1] / spec.adapter_path
        model.load_state_dict(torch.load(str(weights), map_location=device))
        model.eval()
    preprocess = build_preprocess(spec.model_name)
    return model, preprocess


def _load_hf(spec: EmbedderSpec, device: str):
    """transformers CLIP vision tower (+ optional LoRA adapters) — the `hf` backend.

    Returns the SAME (model, preprocess) contract as the open_clip path so the rest of
    ImageEmbedder is backend-agnostic. The ONE difference we keep visible (the forward
    call) lives in `_encode`; the annoying shape difference (HF's extra batch dim) is
    flattened HERE, at the boundary, so the hot loop never branches on backend.
    """
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # noqa: F401

    # TODO(you): build the HF model + processor (the two lines you're here to learn).
    #   model = CLIPVisionModelWithProjection.from_pretrained(spec.model_name).to(device).eval()
    #   processor = CLIPImageProcessor.from_pretrained(spec.model_name)
    # WHY WithProjection: it includes the 512-d visual_projection head, so its
    # `.image_embeds` matches open_clip's encode_image width AND space. Plain
    # CLIPVisionModel would hand back the 768-d pre-projection tower output — wrong.
    model = CLIPVisionModelWithProjection.from_pretrained(spec.model_name).to(device).eval()     # TODO(you)
    processor = CLIPImageProcessor.from_pretrained(spec.model_name)  # TODO(you)

    # LoRA adapters: only when a checkpoint is registered. A relative adapter_path
    # is resolved against the src package root (where train_lora.py writes), so
    # loading works no matter what cwd the caller runs from.
    if spec.adapter_path:
        from peft import PeftModel
        adapter = Path(spec.adapter_path)
        if not adapter.exists():
            adapter = Path(__file__).resolve().parents[1] / spec.adapter_path
        model = PeftModel.from_pretrained(model, str(adapter)).to(device).eval()

    # Normalize preprocess to the open_clip contract: PIL -> [3,H,W] tensor. HF's
    # processor returns a dict whose pixel_values are [1,3,H,W]; squeeze the batch dim
    # so torch.stack in _encode behaves identically for both backends.
    def preprocess(img):
        return processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

    return model, preprocess


class ImageEmbedder:
    """Load CLIP once, embed many images into unit (cosine-space) vectors.

    Loading CLIP per image would be wasteful; construct one of these and reuse it
    (the build job holds one; the photo_id tool holds one).
    """

    def __init__(self, ver: str, device: str | None = None):
        self.ver = ver                       # the identity stamped on every vector
        self.spec = resolve_spec(ver)        # which model/weights this tag means
        self.backend = self.spec.backend     # "open_clip" | "hf" — selects the encode call
        self.dim = self.spec.dim             # expected embedding width
        self.device = device or pick_device()
        self.model, self.preprocess = load_model(self.spec, self.device)
        log.info("ImageEmbedder ready (device=%s, ver=%s)", self.device, self.ver)

    def _encode(self, tensors: list) -> list[list[float]]:
        """One forward pass over a stacked batch → L2-normalized rows."""
        batch = torch.stack(tensors).to(self.device)  # [N, 3, H, W]
        with torch.no_grad():  # inference: no gradients → faster, less memory
            # The one conceptual difference between backends, kept visible on purpose:
            # open_clip exposes encode_image(); HF returns image_embeds off the output.
            if self.backend == "hf":
                feats = self.model(pixel_values=batch).image_embeds  # → [N, dim]
            elif self.backend == "timm":
                feats = self.model(batch)  # ReIDModel.forward            → [N, dim]
            else:
                feats = self.model.encode_image(batch)  # open_clip (today)        → [N, dim]
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
                img = _open_rgb(p)
                tensors.append(self.preprocess(img))  # model's own transform → tensor
                kept.append(p)
            except Exception as e:
                log.warning("unreadable %s: %s", p, e)
        if not tensors:
            return []
        return list(zip(kept, self._encode(tensors)))

    def embed_one(self, path) -> list[float]:
        """Embed a single image (local path or http(s) URL) → one unit vector (raises if unreadable)."""
        img = _open_rgb(path)
        return self._encode([self.preprocess(img)])[0]
