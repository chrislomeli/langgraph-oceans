"""train_arcface.py — v3 trainer: EfficientNetV2-S + ArcFace fluke re-ID.

The successor to train_lora.py (SupCon on CLIP). Differences:
  * backbone is EffNetV2-S @384 (the resolution lever), fully fine-tuned;
  * loss is ArcFace — a CLASSIFICATION loss over the train identities, so it needs
    identity LABELS, not pairs → the PK sampler is gone, just shuffle all images;
  * the ArcFace head is TRAIN-ONLY; we save only the ReIDModel (the embedding path),
    which ImageEmbedder loads as `effnetv2s-arcface-v3`.

Reuses the val-probe + tolerant decode from train_lora, and the whole eval rig downstream.

    uv run python -m training.train_arcface                       # full run
    uv run python -m training.train_arcface --max-whales 50 --steps 400   # sanity
"""

from __future__ import annotations

import argparse
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import get_settings
from models.image_embedder import pick_device
from models.reid_model import DEFAULT_BACKBONE, ReIDModel, build_preprocess
from training.train_lora import DECODE_THREADS, build_probe, decode, load_pool

log = logging.getLogger(__name__)

# ── knobs ──────────────────────────────────────────────────────────────────────────
OUT_VER = "effnetv2s-arcface-v3"
OUT_PATH = Path(__file__).resolve().parents[1] / "artifacts/reid" / f"{OUT_VER}.pt"
EMB_DIM = 512
IMAGE_SIZE = 384
MIN_PHOTOS = 2

LR_BACKBONE = 1e-4          # low: gently adapt ImageNet features
LR_HEAD = 1e-3             # higher: the pool/bn/fc + ArcFace head start from scratch
LR_MIN = 1e-5
ARC_S = 30.0              # ArcFace scale
ARC_M = 0.5              # ArcFace angular margin (radians)

PROBE_EVERY = 250
SEED = 0


class ArcFace(nn.Module):
    """Additive angular margin head (train-only). Pushes an angular GAP between identities.

    logit_y = s·cos(θ_y + m)   for the true class;   logit_j = s·cos(θ_j)  otherwise.
    cos comes from L2-normalized embedding · L2-normalized class prototype.
    """

    def __init__(self, emb_dim: int, n_classes: int, s: float = ARC_S, m: float = ARC_M):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_normal_(self.W)
        self.s, self.m = s, m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.normalize(emb) @ F.normalize(self.W).t()           # [N, C]
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.m)                         # margin on the true angle
        onehot = F.one_hot(labels, num_classes=self.W.size(0)).float()
        return self.s * (onehot * target + (1 - onehot) * cos)


@torch.no_grad()
def probe_reid1(model, device, q_px: torch.Tensor, g_px: torch.Tensor) -> float:
    """Top-1 NN accuracy on the val probe (ReIDModel forward → embedding)."""
    model.eval()
    def emb(px):
        out = []
        for i in range(0, len(px), 64):
            f = model(px[i : i + 64].to(device))
            out.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(out)
    q, g = emb(q_px), emb(g_px)
    hits = (q @ g.t()).argmax(dim=1) == torch.arange(len(q), device=device)
    model.train()
    return hits.float().mean().item()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="EffNetV2-S + ArcFace fluke re-ID trainer")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=200, help="steps with the backbone FROZEN")
    ap.add_argument("--max-whales", type=int, default=0, help="0=all train whales; >0=subset (sanity)")
    args = ap.parse_args()

    get_settings().apply_hf()
    rng = random.Random(SEED)
    device = pick_device()

    # data: every train image as (path, class_index)
    pool = load_pool(args.max_whales or None, MIN_PHOTOS)
    whale_ids = sorted(pool)
    cls = {w: i for i, w in enumerate(whale_ids)}
    items = [(p, cls[w]) for w in whale_ids for p in pool[w]]
    n_classes = len(whale_ids)
    log.info("train: %d images, %d identities (classes)", len(items), n_classes)

    preprocess = build_preprocess(DEFAULT_BACKBONE, IMAGE_SIZE)   # eval transform (no aug; aug is a later lever)
    model = ReIDModel(DEFAULT_BACKBONE, EMB_DIM, pretrained=True).to(device)
    arc = ArcFace(EMB_DIM, n_classes).to(device)
    decoder = ThreadPoolExecutor(max_workers=DECODE_THREADS)

    def sample_batch():
        chosen = [items[rng.randrange(len(items))] for _ in range(args.batch)]
        paths, labels = zip(*chosen)
        tensors = list(decoder.map(lambda p: decode(p, preprocess), paths))
        kept = [(t, lab) for t, lab in zip(tensors, labels) if t is not None]
        tens, labs = zip(*kept)
        return torch.stack(list(tens)).to(device), torch.tensor(labs, device=device)

    query_px, gallery_px = build_probe(rng, preprocess)
    log.info("probe: %d val whales (1 query + 1 gallery each)", len(query_px))

    head_params = (
        list(arc.parameters()) + [model.pool.p]
        + list(model.bn1.parameters()) + list(model.fc.parameters()) + list(model.bn2.parameters())
    )
    opt = torch.optim.Adam(
        [{"params": model.backbone.parameters(), "lr": LR_BACKBONE},
         {"params": head_params, "lr": LR_HEAD}]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=LR_MIN)

    # head warmup: freeze the backbone so the random head/ArcFace stabilize before
    # they perturb the pretrained features.
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    frozen = True

    best = -1.0
    log.info("probe reid@1 BEFORE training: %.3f", probe_reid1(model, device, query_px, gallery_px))

    for step in range(args.steps):
        if frozen and step >= args.warmup:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            frozen = False
            log.info("step %4d  unfroze backbone", step)

        model.train()
        pixels, labels = sample_batch()
        emb = model(pixels)
        logits = arc(emb, labels)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 20 == 0:
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            log.info("step %4d  loss=%.4f  train_acc=%.3f  lr=%.2e%s",
                     step, loss.item(), acc, sched.get_last_lr()[0], "  [frozen]" if frozen else "")

        if (step + 1) % PROBE_EVERY == 0 or step == args.steps - 1:
            score = probe_reid1(model, device, query_px, gallery_px)
            marker = ""
            if score > best:
                best = score
                OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), OUT_PATH)   # ReIDModel only (no ArcFace head)
                marker = "  ← saved (best)"
            log.info("step %4d  PROBE reid@1 = %.3f  (best %.3f)%s", step, score, best, marker)

    log.info("done. best probe reid@1 = %.3f; weights → %s (ver %s)", best, OUT_PATH, OUT_VER)


if __name__ == "__main__":
    main()
