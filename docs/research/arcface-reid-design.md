# Design note — v3: EfficientNetV2-S + ArcFace fluke re-ID

*Status: design, not built. The successor to the CLIP+LoRA line. Mid-level — interfaces
and build order, not code.*

## Why this exists (one paragraph)
The CLIP+LoRA line topped out at val reid@1 ≈ 0.27. Two probes told us why: cropping is
already done by the data (not a lever), and **resolution is the lever** (stock CLIP at 336
vs 224 gave +60% relative, while bigger-model-same-res gave nothing). CLIP ViT-B/32 is
locked at 224. So we move to a backbone that takes higher resolution and train it with the
re-ID standard loss. Target: clear 0.27 by a wide margin and produce a number that stands
on its own, not just "improved."

## The model (v3)
```
image (384×384)
  → EfficientNetV2-S backbone (timm, ImageNet-pretrained)   # off CLIP, takes any res
  → GeM pool → BN → Linear(→512) → L2-normalize             # the 512-d EMBEDDING (what ships)
  → ArcFace head (512 × C, C=#train identities)             # TRAINING ONLY, discarded after
```
- **Embedding is 512-d on purpose** → reuses the existing `fluke_embeddings vector(512)`
  column. No schema migration (the CLIP-L/14 path would have needed vector(768)).
- **ArcFace head is train-only.** At inference we drop it and do cosine retrieval over the
  512-d embedding — identical to today's photo_id path. The tool/eval don't change.

## ArcFace, just enough
Additive angular margin classification. The 512-d embedding and each class weight are
L2-normalized, so their dot product is cos θ (angle to each identity's prototype). For the
TRUE identity, add a margin to the angle before softmax: logit_y = s·cos(θ_y + m); others
s·cos(θ_j). Cross-entropy on that. Effect: forces an angular GAP between identities →
tighter, better-separated clusters than plain softmax or SupCon. Standard knobs: m≈0.5,
s≈30. C ≈ 11,789 (train whales) → a 512×11,789 weight matrix (cheap).

**Simplification vs SupCon:** ArcFace is a classification loss, so it needs identity LABELS,
not positive/negative PAIRS. The PK sampler goes away — just shuffle all train images with
their identity index. (Class imbalance from 2-photo whales is fine; optionally balance later.)

## What we REUSE (the payoff of earlier discipline)
- **The entire eval rig** — `reid_split`, `ReIDSplitDataset`, `ReIDTask`, `eval.py --split`,
  exact-search gallery scoping. v3 is just a new `embedder_ver`; val/test discipline holds.
- **The val-probe** (`train_lora.build_probe`/`probe_reid1`) — fires during training,
  best-checkpoint logic, our safety net.
- **The embed pipeline** (`embed_images.py --split`) and the catalog re-embed path.
- **scan_corrupt** already cleaned the 4 bad files.

## What's NEW to build
1. `models/reid_model.py` — the EffNetV2-S + GeM + embedding-head module (inference path),
   loadable by `ImageEmbedder` as a new backend (`backend="timm"`), output 512-d.
2. `EmbedderSpec` gains the `timm` backend; register `effnetv2s-arcface-v3`.
3. `training/train_arcface.py` — the ArcFace trainer: identity-label loader, ArcFace head,
   backbone fine-tune (discriminative LR: head high, backbone low), val-probe + checkpoint.
   Reuses the SupCon/probe scaffolding's shape.
4. Resolution: 384 train. The embedder's preprocess must match (resize/crop to 384).

## Training plan (first real run)
- Fine-tune the WHOLE backbone (ImageNet→flukes) at a low LR (~1e-4 backbone, ~1e-3 head),
  cosine decay. ArcFace m=0.5, s=30.
- Epochs over all ~78k train images; val-probe every N steps, save best embedding by probe.
- Watch for: ArcFace not converging (loss flat / probe flat) → lower s or m, or warm up the
  head with the backbone frozen for the first epoch. SupCon is the fallback if it won't move.

## Honest costs / risks
- **Compute**: full backbone fine-tune at 384 on MPS is hours per run, not minutes. The
  realistic local ceiling is EffNetV2-S@384; 512–768px (true Kaggle regime) wants cloud GPU.
- **Three changes at once** (backbone + loss + resolution): if the number is bad we won't
  know which knob — mitigated by the val-probe (see it converge or not) + SupCon fallback.
- **Schema**: none — 512-d output reuses the column. (This is why we chose 512.)
- **Full-catalog re-embed** at 384 with the new model before the final eval.

## Build order
1. Design note (this). →
2. `reid_model.py` inference module + `timm` backend in `ImageEmbedder`; smoke: embeds an
   image, 512-d, unit-norm. (No training yet — de-risks the backbone/plumbing first.)
3. `train_arcface.py`: label loader + ArcFace head + fine-tune loop + probe/checkpoint.
4. Short overfit sanity (small subset) — confirm ArcFace drives the probe up at all.
5. Full train → val eval vs the 0.27 baseline.
6. If good: full-catalog re-embed → freeze → one-shot test. If not: diagnose (head warmup /
   s,m / SupCon fallback) before scaling.
