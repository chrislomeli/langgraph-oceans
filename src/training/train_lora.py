"""train_lora.py — the real LoRA fine-tune (viability-spike sized).

Same loop you saw in lora_raw.py, but on REAL train whales from datasets.reid_split,
with a proper metric loss (batch-hard triplet) and the trained adapter SAVED to disk.

    reid_split (split='train') ─► PK batches (P whales × K photos)
        ─► CLIP+LoRA embed ─► batch-hard triplet loss ─► nudge adapter ─► repeat
        ─► save adapter to artifacts/lora/<OUT_VER>/

Start small (N_WHALES=30) to prove LoRA moves the needle in minutes; scale up later.
Validation / early-stopping is deliberately NOT here yet — that arrives with the eval.

    uv run python -m training.train_lora
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from PIL import Image

from config import get_settings
from data.embed_images import resolve_path
from models.image_embedder import pick_device
from stores.postgres import get_pg_gateway
from training.lora_model import get_lora_model

log = logging.getLogger(__name__)

# ── knobs (start tiny; these are the viability-spike values) ───────────────────────
BASE_VER = "clip-hf-vitb32-v1"                      # the frozen foundation to adapt
OUT_VER = "clip-hf-vitb32-lora-v1"                  # what the saved adapter is called
OUT_DIR = Path("artifacts/lora") / OUT_VER          # where save_pretrained writes

N_WHALES = 30        # size of the train pool we sample from
P = 8                # whales per batch
K = 4                # photos per whale per batch  (batch = P*K = 32 images)
STEPS = 200          # number of nudges
LR = 1e-4            # gentle — too high here dives into the "collapse" basin
TEMP = 0.1           # SupCon temperature (sharpness of the contrast; lower = sharper)
SEED = 0


# ── 1. DATA: pull a small pool of train whales + their photo paths ─────────────────
def load_pool(n_whales: int, k: int) -> dict[int, list[Path]]:
    """{whale_id: [photo paths]} for n_whales train whales that have >= k photos."""
    gw = get_pg_gateway()
    image_root = get_settings().image_root
    whales = gw.fetch_rows(
        "SELECT individual_id FROM reid_split "
        "WHERE split='train' AND n_photos >= %s ORDER BY individual_id LIMIT %s",
        (k, n_whales),
    )
    ids = [w["individual_id"] for w in whales]
    rows = gw.fetch_rows(
        "SELECT individual_id, asset_ref FROM manifest "
        "WHERE status='ok' AND individual_id = ANY(%s)",
        (ids,),
    )
    pool: dict[int, list[Path]] = {i: [] for i in ids}
    for r in rows:
        pool[r["individual_id"]].append(Path(resolve_path(r["asset_ref"], image_root)))
    return {w: ps for w, ps in pool.items() if len(ps) >= k}   # keep only deep-enough whales


# ── 2. BATCH: the PK sampler — P whales, K photos each ─────────────────────────────
def sample_batch(pool: dict[int, list[Path]], p: int, k: int, rng: random.Random):
    """Return (paths, labels) for one PK batch: p whales, k photos per whale."""
    whales = rng.sample(list(pool), p)
    paths, labels = [], []
    for w in whales:
        for path in rng.sample(pool[w], k):
            paths.append(path)
            labels.append(w)
    return paths, labels


# ── 3. LOSS: supervised contrastive (SupCon) ───────────────────────────────────────
def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temp: float) -> torch.Tensor:
    """Pull each anchor toward its same-whale photos, contrasting against EVERY other
    photo in the batch at once.

    For each anchor i:  maximize  exp(sim·i,positive) / Σ_{j≠i} exp(sim·i,j)
                                                          └── ALL other photos ──┘
    The denominator holds every negative, so collapsing can't lower the loss — squeezing
    everything together makes that sum blow UP. THAT is the anti-collapse property the
    single-negative triplet lacked.

    emb: [N, D] unit vectors. temp: smaller = sharper contrast.
    """
    n = emb.size(0)
    sim = emb @ emb.t() / temp                                  # [N,N] scaled cosine
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()    # row-shift: numerical stability
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    same = labels[:, None] == labels[None, :]
    pos_mask = same & ~eye                                      # positives, excluding self

    denom = (torch.exp(sim) * ~eye).sum(dim=1, keepdim=True)    # Σ over all j ≠ i
    log_prob = sim - torch.log(denom + 1e-12)                   # log-softmax over non-self
    # average the log-prob over each anchor's positives, then over anchors
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    return -mean_log_prob_pos.mean()


# ── 4. TRAIN ────────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    get_settings().apply_hf()
    rng = random.Random(SEED)
    device = pick_device()

    model, preprocess = get_lora_model(BASE_VER, device=device)

    pool = load_pool(N_WHALES, K)
    log.info("pool: %d whales, %d photos", len(pool), sum(len(v) for v in pool.values()))

    # preprocess every photo ONCE (the pool is small) → path -> CPU tensor cache
    cache = {p: preprocess(Image.open(p).convert("RGB"))
             for paths in pool.values() for p in paths}

    def embed(paths: list[Path]) -> torch.Tensor:
        pixels = torch.stack([cache[p] for p in paths]).to(device)
        feats = model(pixel_values=pixels).image_embeds
        return feats / feats.norm(dim=-1, keepdim=True)         # unit vectors

    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for step in range(STEPS):
        paths, labels = sample_batch(pool, P, K, rng)
        emb = embed(paths)
        loss = supcon_loss(emb, torch.tensor(labels, device=device), TEMP)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            # watch separation directly: SAME should rise, DIFFERENT should fall.
            # if they move together (esp. both → 1.0) the model is COLLAPSING.
            with torch.no_grad():
                lab = torch.tensor(labels, device=device)
                cos = emb @ emb.t()
                same = lab[:, None] == lab[None, :]
                eye = torch.eye(len(lab), dtype=torch.bool, device=device)
                s = cos[same & ~eye].mean().item()
                d = cos[~same].mean().item()
            log.info("step %3d  loss=%.4f  SAME=%.3f  DIFF=%.3f", step, loss.item(), s, d)

    # ── 5. SAVE just the adapter (a few MB) → registry's adapter_path ──────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    log.info("saved adapter → %s   (register/embed as %s)", OUT_DIR, OUT_VER)


if __name__ == "__main__":
    main()
