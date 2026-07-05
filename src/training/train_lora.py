"""train_lora.py — the real LoRA fine-tune (Part-4 scale; see lora-train-design.md §11).

Same loop that proved viability (v1: 30 whales / 200 steps → val reid@1 2.7×), now at
full capacity: every train whale, bigger SupCon batches, and a periodic val probe that
checkpoints the BEST adapter — so a long run can't end worse than its best moment.

    reid_split (split='train') ─► PK batches (P whales × K photos)
        ─► CLIP+LoRA embed ─► SupCon loss ─► nudge adapter ─► repeat
        ─► every PROBE_EVERY steps: mini val reid@1 ─► save adapter if best

    uv run python -m training.train_lora
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

from core.config import get_settings
from data.embed_images import resolve_path
from models.image_embedder import pick_device
from stores.postgres import get_pg_gateway
from training.lora_model import get_lora_model

log = logging.getLogger(__name__)

# ── knobs (Part-4 scale values; v1 viability values in the design note table) ──────
BASE_VER = "clip-hf-vitb32-v1"                      # the frozen foundation to adapt
OUT_VER = "clip-hf-vitb32-lora-v2"                  # what the saved adapter is called
# anchored to src/ so training and ImageEmbedder agree on the path regardless of cwd
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts/lora" / OUT_VER

N_WHALES = None      # train-pool size; None = every train whale with >= MIN_PHOTOS
MIN_PHOTOS = 2       # the split's floor: >=2 photos = at least one positive pair
P = 16               # whales per batch
K = 4                # photos per whale per batch  (batch = P*K = 64 images)
STEPS = 3000         # number of nudges
LR = 1e-4            # gentle — too high here dives into the "collapse" basin
LR_MIN = 1e-5        # cosine-decay floor over the full run
TEMP = 0.1           # SupCon temperature (sharpness of the contrast; lower = sharper)
SEED = 0

PROBE_EVERY = 250    # steps between val probes (each probe ≈ one extra batch of work ×12)
PROBE_WHALES = 200   # fixed val whales in the probe: 1 query + 1 gallery photo each
DECODE_THREADS = 8   # per-batch JPEG decode pool (PIL releases the GIL in C code)


# ── 1. DATA: pull the train pool of whales + their photo paths ─────────────────────
def load_pool(n_whales: int | None, min_photos: int, split: str = "train") -> dict[int, list[Path]]:
    """{whale_id: [photo paths]} for split whales with >= min_photos photos.

    n_whales=None takes the whole split — the Part-4 pool. The LIMIT path stays
    for quick smoke runs.
    """
    gw = get_pg_gateway()
    image_root = get_settings().image_root
    limit_clause = "LIMIT %s" if n_whales else ""
    whales = gw.fetch_rows(
        f"SELECT individual_id FROM reid_split "
        f"WHERE split=%s AND n_photos >= %s ORDER BY individual_id {limit_clause}",
        (split, min_photos, n_whales) if n_whales else (split, min_photos),
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
    return {w: ps for w, ps in pool.items() if len(ps) >= min_photos}


# ── 2. BATCH: the PK sampler — P whales, K photos each ─────────────────────────────
def sample_batch(pool: dict[int, list[Path]], whale_ids: list[int], p: int, k: int, rng: random.Random):
    """Return (paths, labels) for one PK batch: p whales, k photos per whale.

    Whales with fewer than k photos are sampled WITH replacement — a duplicated
    positive is a weaker signal than k distinct photos, but it keeps the 2–3-photo
    majority of the catalog usable instead of excluded.
    """
    whales = rng.sample(whale_ids, p)
    paths, labels = [], []
    for w in whales:
        photos = pool[w]
        chosen = rng.sample(photos, k) if len(photos) >= k else rng.choices(photos, k=k)
        for path in chosen:
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


# ── DECODE: tolerant JPEG → tensor (the catalog has a few corrupt files) ───────────
# A handful of the 78k downloaded JPEGs are truncated/unreadable. One bad file must
# not crash a multi-thousand-step run, so decode returns None on failure and callers
# drop it. Known-bad paths are remembered so we don't re-open (and re-log) them.
_BAD_PATHS: set[Path] = set()


def decode(path: Path, preprocess):
    """Path → preprocessed CPU tensor, or None if the file can't be read."""
    if path in _BAD_PATHS:
        return None
    try:
        return preprocess(Image.open(path).convert("RGB"))
    except Exception as e:  # noqa: BLE001 — any decode failure → skip this image
        _BAD_PATHS.add(path)
        log.warning("skipping unreadable image %s (%s)", path, e)
        return None


# ── 4. PROBE: a cheap in-memory proxy of the real val eval ─────────────────────────
def build_probe(rng: random.Random, preprocess) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed probe set: PROBE_WHALES val whales, 1 query + 1 gallery photo each.

    Returns (query_pixels, gallery_pixels) as preprocessed CPU tensors, cached once
    (~2·PROBE_WHALES images — small). reid@1 on this = "of the probe whales, whose
    gallery photo is nearest each query?" — the same question as the real eval,
    minus the full gallery depth. Val only; test stays sealed.
    """
    pool = load_pool(None, 2, split="val")
    whales = rng.sample(sorted(pool), min(PROBE_WHALES, len(pool)))
    queries, galleries = [], []
    for w in whales:
        q, g = rng.sample(pool[w], 2)
        qt, gt = decode(q, preprocess), decode(g, preprocess)
        if qt is not None and gt is not None:   # both halves must be readable
            queries.append(qt)
            galleries.append(gt)
    return torch.stack(queries), torch.stack(galleries)


@torch.no_grad()
def probe_reid1(model, device, query_px: torch.Tensor, gallery_px: torch.Tensor) -> float:
    """Top-1 nearest-neighbor accuracy: query i should hit gallery i."""
    model.eval()
    def emb(px):
        out = []
        for i in range(0, len(px), 64):
            f = model(pixel_values=px[i : i + 64].to(device)).image_embeds
            out.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(out)
    q, g = emb(query_px), emb(gallery_px)
    hits = (q @ g.t()).argmax(dim=1) == torch.arange(len(q), device=device)
    model.train()
    return hits.float().mean().item()


# ── 5. TRAIN ────────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    get_settings().apply_hf()
    rng = random.Random(SEED)
    device = pick_device()

    model, preprocess = get_lora_model(BASE_VER, device=device)

    pool = load_pool(N_WHALES, MIN_PHOTOS)
    whale_ids = sorted(pool)
    log.info("pool: %d whales, %d photos", len(pool), sum(len(v) for v in pool.values()))

    # 78k photos won't fit in a preprocessed-tensor cache (~47 GB), so decode per
    # batch instead, fanning the JPEG work across threads.
    decoder = ThreadPoolExecutor(max_workers=DECODE_THREADS)

    def embed(paths: list[Path], labels: list[int]) -> tuple[torch.Tensor, list[int]]:
        """Decode + embed a batch, dropping unreadable images (labels stay aligned)."""
        tensors = list(decoder.map(lambda p: decode(p, preprocess), paths))
        kept = [(t, lab) for t, lab in zip(tensors, labels) if t is not None]
        tens, labs = zip(*kept)
        pixels = torch.stack(list(tens)).to(device)
        feats = model(pixel_values=pixels).image_embeds
        return feats / feats.norm(dim=-1, keepdim=True), list(labs)   # unit vectors

    query_px, gallery_px = build_probe(rng, preprocess)
    log.info("probe: %d val whales (1 query + 1 gallery each)", len(query_px))

    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR_MIN)
    best = -1.0

    base_probe = probe_reid1(model, device, query_px, gallery_px)
    log.info("probe reid@1 BEFORE training: %.3f", base_probe)

    for step in range(STEPS):
        paths, labels = sample_batch(pool, whale_ids, P, K, rng)
        emb, labels = embed(paths, labels)   # labels re-bound: dropped images removed
        loss = supcon_loss(emb, torch.tensor(labels, device=device), TEMP)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
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
            log.info("step %4d  loss=%.4f  SAME=%.3f  DIFF=%.3f  lr=%.2e",
                     step, loss.item(), s, d, sched.get_last_lr()[0])

        # ── checkpoint on the probe: keep the BEST adapter, not the last one ──────
        if (step + 1) % PROBE_EVERY == 0 or step == STEPS - 1:
            score = probe_reid1(model, device, query_px, gallery_px)
            marker = ""
            if score > best:
                best = score
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(OUT_DIR)
                marker = "  ← saved (best)"
            log.info("step %4d  PROBE reid@1 = %.3f  (best %.3f)%s", step, score, best, marker)

    log.info("done. best probe reid@1 = %.3f (vs %.3f untrained); adapter → %s (ver %s)",
             best, base_probe, OUT_DIR, OUT_VER)


if __name__ == "__main__":
    main()
