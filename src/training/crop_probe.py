"""crop_probe.py — does cropping to the fluke move re-ID? (the top lever, measured)

Holds the embedder FIXED (v2) and varies only WHAT IT SEES: the whole image vs a
fluke crop from a zero-shot detector (OWL-ViT, prompted "a whale fluke / tail").
Same sample of val whales, same gallery, same metric — so any delta is purely framing.

Deliberately a LOWER BOUND: v2 was trained on whole images, so feeding it crops is
off-distribution; a model retrained on crops would do better. If even this lifts
reid@1, the input pipeline is the bottleneck and the detect→crop→retrain project is
justified. If it barely moves, capacity is the bottleneck, not framing.

    uv run python -m training.crop_probe

Reads reid_split val; embeds a sample two ways; prints reid@1 / reid@5 side by side.
Writes nothing to the catalog.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from PIL import Image

from config import get_settings
from data.embed_images import resolve_path
from models.image_embedder import ImageEmbedder, pick_device
from stores.postgres import get_pg_gateway

log = logging.getLogger(__name__)

VER = "clip-hf-vitb32-lora-v2"   # the FIXED embedder; only the input framing changes
N_WHALES = 150                   # val whales sampled (query + gallery each)
K_GALLERY = 6                    # cap gallery photos per whale (bounds detector cost)
SEED = 0

DETECTOR = "google/owlvit-base-patch32"
PROMPTS = ["a whale fluke", "a whale tail", "a whale"]  # broad→narrow; best box wins
BOX_PAD = 0.08                   # pad the crop a touch so we don't shave the edges
EMBED_BATCH = 64


# ── 1. sample: val whales → {whale: (query_path, [gallery_paths])} ─────────────────
def load_sample(rng: random.Random) -> dict[int, tuple[str, list[str]]]:
    gw = get_pg_gateway()
    root = get_settings().image_root
    whales = gw.fetch_rows(
        "SELECT individual_id, query_sighting_id FROM reid_split "
        "WHERE split='val' AND query_sighting_id IS NOT NULL ORDER BY individual_id"
    )
    chosen = rng.sample(whales, min(N_WHALES, len(whales)))
    ids = [w["individual_id"] for w in chosen]
    rows = gw.fetch_rows(
        "SELECT individual_id, sighting_id, asset_ref FROM manifest "
        "WHERE status='ok' AND individual_id = ANY(%s)",
        (ids,),
    )
    by_whale: dict[int, dict[int, str]] = {i: {} for i in ids}
    for r in rows:
        by_whale[r["individual_id"]][r["sighting_id"]] = resolve_path(r["asset_ref"], root)

    sample: dict[int, tuple[str, list[str]]] = {}
    for w in chosen:
        wid, qsid = w["individual_id"], w["query_sighting_id"]
        photos = by_whale[wid]
        if qsid not in photos or len(photos) < 2:
            continue
        qpath = photos[qsid]
        gallery = [p for sid, p in photos.items() if sid != qsid][:K_GALLERY]
        if gallery:
            sample[wid] = (qpath, gallery)
    return sample


# ── 2. detector: crop each image to the best fluke box (fallback = whole) ───────────
class FlukeCropper:
    def __init__(self, device: str):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.device = device
        self.proc = OwlViTProcessor.from_pretrained(DETECTOR)
        self.model = OwlViTForObjectDetection.from_pretrained(DETECTOR).to(device).eval()
        self.fallbacks = 0

    @torch.no_grad()
    def crop(self, img: Image.Image) -> Image.Image:
        inputs = self.proc(text=[PROMPTS], images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        sizes = torch.tensor([img.size[::-1]], device=self.device)  # (h, w)
        res = self.proc.post_process_grounded_object_detection(out, threshold=0.0, target_sizes=sizes)[0]
        if len(res["scores"]) == 0:
            self.fallbacks += 1
            return img
        x0, y0, x1, y1 = res["boxes"][res["scores"].argmax()].tolist()
        w, h = img.size
        pad_x, pad_y = (x1 - x0) * BOX_PAD, (y1 - y0) * BOX_PAD
        box = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:   # degenerate box → whole image
            self.fallbacks += 1
            return img
        return img.crop(box)


# ── 3. embed + score ────────────────────────────────────────────────────────────────
def embed_images(embedder: ImageEmbedder, imgs: list[Image.Image]) -> np.ndarray:
    """PIL images → unit-vector matrix, reusing the embedder's own preprocess+encode."""
    vecs: list[list[float]] = []
    for i in range(0, len(imgs), EMBED_BATCH):
        tensors = [embedder.preprocess(im) for im in imgs[i : i + EMBED_BATCH]]
        vecs.extend(embedder._encode(tensors))
    return np.asarray(vecs, dtype=np.float32)


def score(q_vecs, q_labels, g_vecs, g_labels) -> tuple[float, float]:
    """reid@1 / reid@5: rank gallery IMAGES by cosine, collapse to best-per-whale."""
    g_labels = np.asarray(g_labels)
    hit1 = hit5 = 0
    for qv, ql in zip(q_vecs, q_labels):
        sims = g_vecs @ qv                       # unit vectors → dot = cosine
        best: dict[int, float] = {}
        for s, gl in zip(sims, g_labels):
            if gl not in best or s > best[gl]:
                best[gl] = float(s)
        ranked = sorted(best, key=best.get, reverse=True)
        if ranked and ranked[0] == ql:
            hit1 += 1
        if ql in ranked[:5]:
            hit5 += 1
    n = len(q_labels)
    return hit1 / n, hit5 / n


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    get_settings().apply_hf()
    rng = random.Random(SEED)
    device = pick_device()

    sample = load_sample(rng)
    q_labels = list(sample)
    q_paths = [sample[w][0] for w in q_labels]
    g_labels, g_paths = [], []
    for w in q_labels:
        for p in sample[w][1]:
            g_labels.append(w)
            g_paths.append(p)
    log.info("sample: %d whales, %d query + %d gallery images", len(q_labels), len(q_paths), len(g_paths))

    # decode once (tolerant); keep labels aligned to successfully-opened images
    def decode_all(paths, labels):
        imgs, labs = [], []
        for p, lab in zip(paths, labels):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                labs.append(lab)
            except Exception as e:  # noqa: BLE001
                log.warning("skip unreadable %s (%s)", p, e)
        return imgs, labs

    q_imgs, q_labels = decode_all(q_paths, q_labels)
    g_imgs, g_labels = decode_all(g_paths, g_labels)

    embedder = ImageEmbedder(ver=VER, device=device)

    # ── ARM A: whole images ────────────────────────────────────────────────────────
    log.info("embedding WHOLE images…")
    qv_whole = embed_images(embedder, q_imgs)
    gv_whole = embed_images(embedder, g_imgs)
    r1_whole, r5_whole = score(qv_whole, q_labels, gv_whole, g_labels)

    # ── ARM B: fluke crops ─────────────────────────────────────────────────────────
    log.info("detecting + cropping flukes (OWL-ViT)…")
    cropper = FlukeCropper(device)
    q_crops = [cropper.crop(im) for im in q_imgs]
    g_crops = [cropper.crop(im) for im in g_imgs]
    log.info("embedding CROPPED images… (detector fell back to whole on %d/%d)",
             cropper.fallbacks, len(q_imgs) + len(g_imgs))
    qv_crop = embed_images(embedder, q_crops)
    gv_crop = embed_images(embedder, g_crops)
    r1_crop, r5_crop = score(qv_crop, q_labels, gv_crop, g_labels)

    # ── verdict ────────────────────────────────────────────────────────────────────
    n_gallery = len(set(g_labels))
    print(f"\n===== CROP PROBE ({VER}, {len(q_labels)} queries vs {n_gallery}-whale gallery) =====")
    print(f"  {'arm':14s}  reid@1   reid@5")
    print(f"  {'whole image':14s}  {r1_whole:.3f}    {r5_whole:.3f}")
    print(f"  {'fluke crop':14s}  {r1_crop:.3f}    {r5_crop:.3f}")
    print(f"  {'Δ reid@1':14s}  {r1_crop - r1_whole:+.3f}")
    print(f"  detector fallback-to-whole rate: {cropper.fallbacks}/{len(q_imgs)+len(g_imgs)}")
    print("  (lower bound: v2 was trained on whole images; a crop-retrained model would do better)")


if __name__ == "__main__":
    main()
