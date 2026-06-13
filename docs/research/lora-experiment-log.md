# LoRA re-ID experiment log — what actually moved the number

A running record of every change tried to drive the re-identification metric up
(and the training loss down without collapse). Newest at the bottom. The honest
yardstick throughout is **val reid@1** on `datasets.reid_split` (by-individual,
disjoint gallery, exact search) — "can it re-ID a whale it never trained on?"

Baseline to beat: **raw CLIP, val reid@1 = 0.025 / reid@5 = 0.054 / MRR = 0.035.**

| # | change | why | result |
|---|---|---|---|
| 1 | batch-hard **triplet** loss | first metric-learning attempt | **COLLAPSED** — SAME & DIFF cosine both → 1.0; loss stuck at the margin. Tell: SAME−DIFF gap stops widening. Cause: triplet pulls many positives but pushes only ONE negative. |
| 2 | switch to **SupCon** loss | contrasts each anchor against ALL negatives at once, so collapse *raises* the loss | clean separation; SAME−DIFF gap 0.006 → 0.67 on the tiny set |
| 3 | **viability run** (v1): 30 whales, P8×K4, 200 steps, LR 1e-4, temp 0.1 | prove the lift exists before scaling | **val reid@1 0.025 → 0.067 (2.7×)**, reid@5 0.054 → 0.146, MRR 0.035 → 0.095 |
| 4 | **scale run** (v2): all 11,789 train whales, P16×K4 (64/batch), 3000 steps, cosine LR 1e-4→1e-5, best-probe checkpoint (peak @ step 1749) | turn the viable recipe up to full capacity | **val reid@1 0.067 → 0.277 (~4× over v1, ~11× over baseline)**, reid@5 0.146 → 0.454, MRR 0.095 → 0.345 |

**Decision: stop here, do NOT escalate to ArcFace.** v2 clears the bar by a wide margin
(27.7% top-1 on unseen whales vs 2.5% raw CLIP). ArcFace/BioCLIP stay in reserve; the
number didn't ask for them. v2 is the frozen final adapter.

## Knobs and why they're set where they are
- **Loss = SupCon, temp 0.1.** The anti-collapse property is the whole reason (see #1→#2).
- **PK sampler, P×K batches.** Every batch must contain multiple photos of multiple
  whales, or SupCon has no positives/negatives to contrast. Bigger P = more in-batch
  negatives = stronger signal (P went 8→16 from v1→v2).
- **Sample-with-replacement for <K-photo whales** (v2). Keeps the 2–3-photo majority of
  the catalog usable instead of excluded; a duplicated positive is a weak signal but
  better than dropping the whale.
- **Cosine LR decay 1e-4 → 1e-5** (v2). A long run wants to anneal; the late settling
  is visible in the probe (gains resumed after a mid-run plateau).
- **Best-probe checkpoint, not last** (v2). A 3000-step run must not end worse than its
  best moment; a 200-whale val probe every 250 steps saves the adapter only on a new best.
- **Tolerant decode** (v2). A few corrupt JPEGs in the 78k train images crashed the run;
  decode now skips + remembers them. Dataset-level fix: `data/scan_corrupt.py` scanned all
  114,126 images and flagged the **4** unreadable ones `status='corrupt'`, so every consumer
  (train/embed/eval) excludes them — they all already filter `status='ok'`.

## v2 probe trajectory (200-whale proxy, not the real eval)
Untrained: 0.070
- step  249 → 0.260
- step  499 → 0.300
- step  749 → 0.305
- step  999 → 0.355
- step 1249 → 0.380
- step 1499 → 0.390
- step 1749 → **0.395 (peak — this is the saved adapter)**
- step 1999 → 0.375 · 2249 → 0.395 · 2499 → 0.380 · 2749 → 0.390 · 2999 → 0.390
Plateaued ~0.39 after step ~1500; best-checkpoint kept the step-1749 adapter. The probe
(200 whales, 1+1) reads lower than the full eval (full gallery depth) — it's a relative
signal for checkpointing, not the headline.

## Final eval (full val, 1,188 unseen whales)
| ver | reid@1 | reid@5 | MRR |
|---|---|---|---|
| baseline (raw CLIP) | 0.025 | 0.054 | 0.035 |
| v1 (viability) | 0.067 | 0.146 | 0.095 |
| **v2 (scaled)** | **0.277** | **0.454** | **0.345** |
| **v2 on test** (one-shot, 1,136 whales) | **0.273** | **0.460** | **0.342** |

Test ≈ val (0.273 vs 0.277) → **no overfit to val; the lift generalizes.** v2 is final.

## Crop probe (2026-06-12) — the assumed top lever is ALREADY SPENT
Hypothesis: framing is the bottleneck — embedding whole scenes wastes CLIP's 224px on
ocean. Tested whole vs fluke-crop (OWL-ViT) on 150 val whales, same v2 embedder.
- **Result: no lift** (reid@1 0.527 whole → 0.520 crop; gallery=150 so absolutes are high).
- **But the detector failed**: 0/909 fallbacks at threshold 0.0, and inspected crops cover
  85–100% of the frame — OWL-ViT can't localize "whale fluke" (out of vocab), so crop≈whole.
- **The real finding, confirmed by EYE**: the catalog photos are already tight fluke
  close-ups (tail fills 80–100% of frame). This is a curated fluke-ID dataset, not wild
  scenes — so cropping is already done for us. **Framing is NOT the lever here.**
- Lesson: look at the data before importing a playbook. The Kaggle "crop first" assumption
  doesn't transfer because their raw data and ours differ at the source.

## Re-ranked levers FOR THIS DATASET (post-inspection)
1. ~~Crop to fluke~~ — already done by the data; nothing to gain.
2. **Resolution** — NEW #1. Flukes are ~1200px; CLIP ViT-B/32 sees 224px, blurring the
   fine trailing-edge notches/scars that carry identity. Needs a backbone that takes
   higher res (ViT-B/16, CLIP-L/14-336, or non-CLIP) — coupled with #4.
3. **ArcFace** — the loss lever (we're on SupCon).
4. **Stronger backbone** — capacity, and the enabler for higher resolution.
5. **Data cleaning** — drop genuinely bad shots (half-submerged/blurred flukes seen in
   the sample); they can't be matched and only add noise.

So the "best hose" for THIS data is a **higher-resolution backbone + ArcFace**, NOT a
detection pipeline (the data doesn't need one). Arguably a cleaner build.

## Resolution probe (2026-06-12) — resolution is the lever, capacity is not
Disentangled the two coupled knobs with three STOCK (no fine-tune) CLIP backbones on
the same 150-whale val sample (absolutes inflated by the small gallery; deltas are clean):
| backbone | reid@1 | reid@5 |
|---|---|---|
| ViT-B/32 @224 (current) | 0.073 | 0.193 |
| ViT-L/14 @224 (bigger, same res) | 0.067 | 0.167 |
| ViT-L/14 @336 (bigger + higher res) | 0.107 | 0.220 |
- **Capacity effect** (B/32→L/14 @224): **−0.007** — bigger model alone does nothing.
- **Resolution effect** (L/14 @224→@336): **+0.040 (~60% relative)** — on stock weights.
- Verdict: spend the fine-tune budget on **resolution**, not model size. Confirms the
  eyeball finding — identity is in fine detail that 224px blurs away.

## The build that follows: v3 = CLIP ViT-L/14-336 + LoRA (+ ArcFace)
- Higher res = the proven lever; still CLIP so the peft/LoRA wiring transfers (same
  q/k/v/out_proj). Stock already +60% relative; fine-tuning should compound on top.
- **Costs to budget**: (1) L/14-336 is ~3–4× params and 336px ~2.3× pixels → training is
  much slower than B/32; (2) L/14 outputs **768-d**, but fluke_embeddings.embedding is
  vector(512) → needs a 768-d column/table (schema plumbing); (3) full-catalog re-embed.
- **Resolution ladder**: 224 (now) → 336 (CLIP-L/14-336, easy, peft-able) → 512–768
  (needs a non-CLIP backbone like ConvNeXt/EfficientNet — the Kaggle regime, a bigger build).
- ArcFace is the orthogonal loss lever to stack once the backbone is in place.

## v3 build (2026-06-12) — off CLIP: EfficientNetV2-S + ArcFace @384
Decision: stop the incremental ladder, build the real thing (user's call). Design in
`arcface-reid-design.md`. Progress:
- **Plumbing** (`models/reid_model.py` + `timm` backend in ImageEmbedder): smoke PASSED —
  effnetv2s-arcface-v3 embeds a 512-d unit vector (reuses vector(512), no schema change).
- **Trainer** (`training/train_arcface.py`): ArcFace head (s=30, m=0.5) over 11,789 train
  identities; discriminative LR (backbone 1e-4 / head 1e-3); 300-step backbone warmup;
  reuses the val-probe + tolerant decode; saves ReIDModel only (ArcFace head discarded).
- **Sanity** (50 whales, 400 steps): train_acc 0.00 → 0.50 → **0.94**, loss 18.7 → 0.33,
  clean unfreeze, no NaN → ArcFace machinery confirmed. (val probe ~0.07, expected — only
  50 train whales; train_acc is the sanity metric.)
- **Full run**: 6000 steps, all identities, ~1h on M3. Probe 0.030 → **0.570** peak (vs CLIP-v2's
  0.395). Full val eval (1,188 unseen whales):

| ver | reid@1 | reid@5 | MRR |
|---|---|---|---|
| raw CLIP | 0.025 | 0.054 | 0.035 |
| CLIP-v2 LoRA | 0.277 | 0.454 | 0.345 |
| **v3 EffNetV2-S + ArcFace @384** | **0.627** | **0.749** | **0.674** |

**~25× the original raw-CLIP MRR (0.024 → 0.674), ~2.3× over CLIP-v2.** A genuinely good
single-model re-ID result — the resolution+ArcFace+real-backbone bet, exactly as the probes
predicted. v3 is the frozen final model.

**One-shot TEST (1,136 sealed whales): reid@1 0.619 · reid@5 0.759 · MRR 0.672.** Test ≈ val
→ no overfit; it generalizes. This is the reported headline. Weights:
`src/artifacts/reid/effnetv2s-arcface-v3.pt`; registered as `effnetv2s-arcface-v3`.

## Final scoreboard (test, the honest number)
| model | reid@1 | MRR | note |
|---|---|---|---|
| raw CLIP ViT-B/32 | 0.025 | 0.035 | the floor |
| CLIP-B/32 + LoRA + SupCon (v2) | 0.273 | 0.342 | learned LoRA; resolution-capped at 224 |
| **EffNetV2-S @384 + ArcFace (v3)** | **0.619** | **0.672** | the result |

Levers that moved it, in order of impact: **resolution (224→384)** + **ArcFace loss** +
**a real fine-tuned backbone**. Cropping was a non-lever (data already tight). Model size
alone was a non-lever (capacity probe flat). All decided by cheap probes before the build.

## Escalation ladder (if SupCon plateaus below target)
triplet → **SupCon** (here) → **ArcFace** (additive-angular-margin classification head;
the standard next gun for face/animal re-ID). BioCLIP is a *fallback base model*, not a
loss — it's trained for species classification, not individual ID, so it's reserve, not
the obvious win. Don't open with the big gun: only escalate if the number says to.
