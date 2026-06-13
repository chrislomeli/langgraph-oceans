"""resolution_probe.py — is INPUT RESOLUTION the lever for fluke re-ID?

Cropping is already done by the data (see crop_probe). Next hypothesis: CLIP's 224px
window blurs the fine trailing-edge notches/scars that distinguish flukes. Test it by
embedding the SAME val sample with three STOCK (no fine-tune) CLIP backbones:

    ViT-B/32 @224   our baseline backbone
    ViT-L/14 @224   bigger model, SAME resolution   → isolates CAPACITY
    ViT-L/14 @336   bigger model, HIGHER resolution  → isolates RESOLUTION

So B32→L14@224 is the pure capacity effect, and L14@224→L14@336 is the pure resolution
effect. All stock weights, whole images, same whales — each delta is clean. This tells
us WHICH knob to spend the fine-tune on before committing to the big build.

    uv run python -m training.resolution_probe
"""

from __future__ import annotations

import gc
import logging
import random

import numpy as np
import open_clip
import torch
from PIL import Image

from config import get_settings
from data.embed_images import resolve_path
from models.image_embedder import pick_device
from stores.postgres import get_pg_gateway
from training.crop_probe import K_GALLERY, N_WHALES, SEED, load_sample, score

log = logging.getLogger(__name__)

# (label, open_clip arch, pretrained tag) — all OpenAI weights, fair lineage
BACKBONES = [
    ("ViT-B/32 @224", "ViT-B-32", "openai"),
    ("ViT-L/14 @224", "ViT-L-14", "openai"),
    ("ViT-L/14 @336", "ViT-L-14-336", "openai"),
]
EMBED_BATCH = 32


def embed(model, preprocess, imgs, device) -> np.ndarray:
    vecs: list[np.ndarray] = []
    for i in range(0, len(imgs), EMBED_BATCH):
        batch = torch.stack([preprocess(im) for im in imgs[i : i + EMBED_BATCH]]).to(device)
        with torch.no_grad():
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        vecs.extend(f.cpu().numpy())
    return np.asarray(vecs, dtype=np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    get_settings().apply_hf()
    device = pick_device()
    rng = random.Random(SEED)

    sample = load_sample(rng)
    q_labels0 = list(sample)
    q_paths = [sample[w][0] for w in q_labels0]
    g_labels0, g_paths = [], []
    for w in q_labels0:
        for p in sample[w][1]:
            g_labels0.append(w)
            g_paths.append(p)

    def decode_all(paths, labels):
        imgs, labs = [], []
        for p, lab in zip(paths, labels):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                labs.append(lab)
            except Exception as e:  # noqa: BLE001
                log.warning("skip unreadable %s (%s)", p, e)
        return imgs, labs

    q_imgs, q_labels = decode_all(q_paths, q_labels0)
    g_imgs, g_labels = decode_all(g_paths, g_labels0)
    n_gallery = len(set(g_labels))
    log.info("sample: %d whales, %d query + %d gallery images (N_WHALES=%d, K_GALLERY=%d)",
             len(q_labels), len(q_imgs), len(g_imgs), N_WHALES, K_GALLERY)

    results = []
    for label, arch, pretrained in BACKBONES:
        log.info("loading %s (%s/%s)…", label, arch, pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model = model.to(device).eval()
        qv = embed(model, preprocess, q_imgs, device)
        gv = embed(model, preprocess, g_imgs, device)
        r1, r5 = score(qv, q_labels, gv, g_labels)
        results.append((label, r1, r5))
        log.info("%s → reid@1=%.3f reid@5=%.3f", label, r1, r5)
        del model, preprocess, qv, gv  # free before loading the next big backbone
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    print(f"\n===== RESOLUTION PROBE (STOCK CLIP, {len(q_labels)} queries vs {n_gallery}-whale gallery) =====")
    print(f"  {'backbone':16s}  reid@1   reid@5")
    for label, r1, r5 in results:
        print(f"  {label:16s}  {r1:.3f}    {r5:.3f}")
    b32, l224, l336 = (r[1] for r in results)
    print(f"\n  capacity effect  (B/32→L/14 @224):  {l224 - b32:+.3f} reid@1")
    print(f"  resolution effect (L/14 @224→@336):  {l336 - l224:+.3f} reid@1")
    print("  (all STOCK, no fine-tune — this ranks WHICH knob a fine-tune should spend on)")


if __name__ == "__main__":
    main()
