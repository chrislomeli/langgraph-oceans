# Whale photo-ID re-identification — a case study

How a humpback fluke re-identifier went from **MRR 0.024 → 0.672** (top-1 accuracy
**2.5% → 62%**) on whales it had never seen — and an honest accounting of where it sits
against world-class, what trade-offs got us here, and where it would go next.

> **The one-line pitch:** I didn't brute-force a model. I built an honest measuring rig,
> used cheap experiments to find the *actual* bottleneck (not the one the playbook assumed),
> and made informed trade-offs to land a genuinely strong single-model result on a laptop.

---

## 1. The problem, and why it's hard

Given a photo of a humpback's tail fluke, name the individual — out of a catalog of
~24,000 known whales. This is **fine-grained re-identification**: the classes (individuals)
look nearly identical, the distinguishing marks are small (notches, scars, pigment on the
trailing edge), and — critically — **the whale you're identifying was never in training**.
That last point is the honest version of the problem: not "re-spot a whale you memorized,"
but "recognize one you've never seen," which is how the tool gets used in the real world.

## 2. The yardstick (built before any modeling)

You can't improve what you can't measure honestly, so the first build was the eval, not a
model:

- **Split by *individual*, not by photo.** Whole whales go to train / val / test, so a test
  whale's photos never leak into training. This is the difference between measuring
  memorization and measuring generalization.
- **Gallery scoping.** A val query is matched only against *other val whales* — otherwise a
  query could rank a training whale #1 and silently inflate the score.
- **Exact search, not approximate.** The eval uses exact cosine, not the fast ANN index, so
  the number reflects the *embedding's* quality, not the index's approximation error.
- **val to iterate, test spent once.** Every decision below was made on val; the test set
  was scored exactly one time per final model, at the end.

Metrics: **Recall@1** (top-1 accuracy), **Recall@5**, **MRR** (mean reciprocal rank).

## 3. The journey, blow by blow

| step | what | result (val reid@1) | the lesson |
|---|---|---|---|
| **v0** | Raw CLIP ViT-B/32, no training | 0.025 | The floor. CLIP knows "whale," not *which* whale. |
| **v1** | CLIP + LoRA, batch-hard **triplet** loss | — | **Collapsed** — every embedding → one point. Triplet pushes only one negative per anchor. Tell: the same-vs-different gap stops widening. |
| **v1'** | Swap loss to **SupCon** | 0.067 | Contrasts *all* negatives at once, so collapse *raises* the loss. Viability proven (~3×). |
| **v2** | Scale LoRA: all 11.8k train whales, bigger batches, best-checkpoint | **0.277** | A real lift — but it plateaued. Honest read: **mediocre**, not portfolio-worthy. |
| — | **Stop and diagnose** | — | Refused to ship 0.27. Asked "is this *good*, or just *improved*?" — different questions. |
| **probe A** | Crop to the fluke (zero-shot detector) | no change | Detector failed *and* — looking at the data — the photos were **already tight fluke close-ups**. Cropping was a non-lever. *Lesson: look at the data before importing a playbook.* |
| **probe B** | Resolution vs capacity (stock CLIP B/32@224 vs L/14@224 vs L/14@336) | +60% rel from 224→336; **0** from bigger-model-same-res | **Resolution is the lever; raw model size is not.** Identity lives in fine detail that 224px blurs away. |
| **v3** | Off CLIP → **EfficientNetV2-S @384 + ArcFace** | **0.627** | The result. The probes called it exactly. |

Each probe cost ~10 minutes and was run *before* committing compute. The crop probe saved
us from building a detection pipeline we didn't need; the resolution probe pointed straight
at the backbone swap that worked.

## 4. The result

Final model **`effnetv2s-arcface-v3`** — EfficientNetV2-S backbone @384px, GeM pooling, a
512-d embedding head, trained with **ArcFace** (additive angular-margin) over 11,789 train
identities. ~1 hour to train on an M3 MacBook.

| model | test reid@1 | test reid@5 | test MRR |
|---|---|---|---|
| raw CLIP ViT-B/32 | 0.025 | 0.054 | 0.035 |
| CLIP + LoRA + SupCon (v2) | 0.273 | 0.460 | 0.342 |
| **EffNetV2-S @384 + ArcFace (v3)** | **0.619** | **0.759** | **0.672** |

**val ≈ test** (0.627 vs 0.619) → the lift generalizes; it isn't overfit to the val set
we tuned on. ~25× the original MRR.

## 5. The method — what's transferable beyond whales

The single most important habit: **attack the levers top-down.**

> **input → objective → capacity → polish.**
> *What does the model see* before *how is it taught* before *how big it is* before
> *squeezing the last points.*

We initially over-invested in the middle (the loss, the LoRA scale) while leaving the top
(resolution) untouched — which is exactly why v2 was capped at 0.27. The fix wasn't a bigger
model (the capacity probe proved that does nothing here); it was letting the model *see* the
fine detail. A bigger model can't recover detail that 224px already threw away.

Second habit: **don't trust a playbook on a new dataset.** The Kaggle consensus is "crop
first" — but that's because *their* raw data was wide scenes. Ours was already cropped.
Four eyeballed images overturned an entire assumed work-stream.

## 6. Where this sits vs. world-class — the honest pitch

The reference point is the 2022 Kaggle "Happywhale — Whale and Dolphin Identification"
competition (same data source). Honest framing:

- **World-class on that task scored ~high-0.80s MAP@5.** Note this is *not* an apples-to-
  apples number: it's a different metric (MAP@5 with an open-set "new individual" class),
  multi-species, ~15k individuals, and — decisively — **ensembles of large backbones at
  512–768px trained on multi-GPU compute.**
- **Our v3 is a single model, 384px, ~1 hour on one laptop GPU, reid@1 0.62 on a 1,136-way
  disjoint gallery.** That is a *strong single-model result* — it lands in the range a
  competent single entry reaches before the ensembling-and-scale phase that wins competitions.

So the honest pitch is not "I matched the world record." It's: **"I identified the same
load-bearing levers the winners used (resolution + margin-based metric loss + a real
backbone), reached a solid single-model number with a fraction of the compute, and I can
tell you precisely what the remaining gap to world-class costs."** The trade-offs were
deliberate, not accidental — and being able to *name the gap and its price* is the senior
move, more than the number itself.

## 7. Where I'd go next (and the expected payoff)

Ranked by expected impact per unit effort. We are **not** doing these — the current result
is strong enough for the agentic work that's the actual goal — but this is the roadmap a
reviewer should know we understand:

1. **Higher resolution (512–768px).** The proven lever, pushed further. Requires a backbone
   that's happy at that size and real GPU memory — the biggest single expected gain.
2. **Bigger / better backbone** (EfficientNetV2-M/L, ConvNeXt) — capacity *now pays off*
   because the input is good (it didn't, before, at 224).
3. **Sub-center ArcFace + dynamic margin** — handles within-individual variation and class
   imbalance better than vanilla ArcFace; standard in the winning solutions.
4. **Augmentation** (affine, color, cutout — carefully, no identity-breaking flips) — cheap
   robustness; we ran v3 with none.
5. **Hard-negative mining / re-ranking + test-time augmentation** — last-mile points.
6. **Ensembling** — the final lever competitions are won on; high cost, diminishing returns.

The realistic ceiling on a single laptop is roughly where we are; the items above mostly
want a cloud GPU. **The constraint was compute, and the trade-off — a strong single model
now vs. a marginally better one after days of cloud training — was made knowingly.**

## 8. The part that's better than the number — the engineering seam

The number will date; the system won't. What this project actually demonstrates:

- A **swappable embedder registry** — every model (raw CLIP, HF CLIP, LoRA, EffNetV2+ArcFace)
  is one config tag; A/B is an env var, not a rewrite. v3 dropped in behind the same
  `photo_id` tool with zero downstream changes.
- A **rigorous, leak-proof eval** that was reused verbatim across four very different models —
  the disjoint split, gallery scoping, and val/test discipline never moved.
- A **documented decision trail** (`docs/research/lora-experiment-log.md`): every change, why
  it was tried, and what it produced — including the failures (the triplet collapse, the
  broken crop probe) that taught the most.

That seam is the reason the 25× improvement was *measurable and trustworthy* rather than a
lucky number — and it's the part that says "this person can do ML," not just run it.

---

*Artifacts: weights `src/artifacts/reid/effnetv2s-arcface-v3.pt`; model `src/models/reid_model.py`;
trainer `src/training/train_arcface.py`; eval `src/evals/photo_id/`; full log
`docs/research/lora-experiment-log.md`; design `docs/research/arcface-reid-design.md`.*
