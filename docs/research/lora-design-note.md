# LoRA fine-tune — component design note (roadmap item 13, Phase C)

> Mid-level design: *what we're adding and how it fits.* The goal is a **measured
> accuracy lift** over the raw-CLIP baseline, produced by fine-tuning the image
> embedder — bounded to **one tool's internals**, evaluated **entirely locally**.

Status: design, 2026-06-08. Builds on the completed Phase A (tool + Layer-A eval +
frozen split). Supersedes the LoRA sketch in `photo-id-ml-plan.md` §3 now that the
eval and versioning seams actually exist.

---

## 1. What we're adding

Fine-tune the Layer-5 image embedder (CLIP ViT-B/32) with **LoRA adapters** via
**metric learning**, so same-individual flukes cluster and different whales separate.
Produce a new embedder version, re-embed the catalog, and re-run the existing
Layer-A eval to **measure the lift** over the v1 baseline.

The baseline to beat (raw CLIP, local, deterministic):
```
reid@1 = 0.016    reid@5 = 0.040    MRR = 0.024
```

## 2. Where it sits — almost no new architecture, by design

LoRA is an **offline training process** that produces a new artifact and re-uses
every existing seam. It does **not** touch the tool (Layer 3) or the eval code.

```
NEW  src/training/         offline: train LoRA adapters on fluke images
         │ produces
Layer 5  image_embedder    a v2 variant = base CLIP + LoRA adapter,
         │                 EMBEDDER_VER = "clip-vitb32-lora-v2"
         │ re-embed (reuse embed_images.py)
Layer 4  fluke_embeddings  v2 rows COEXIST with v1 (filter by embedder_ver)
         │ same eval, pointed at v2
Layer A  eval.py (LOCAL)   reid@1/@5/MRR for v2  →  compare to v1   ← the payoff
```

The whole reason this slots in cleanly: **`embedder_ver` is a filter, not a wipe.**
v1 and v2 live side by side; the eval just chooses which version to read. This is
the migration design from `photo-id-ml-plan.md` finally being cashed in.

## 3. The seams — reused, not invented

| Seam | Already exists? | LoRA's use |
|---|---|---|
| `EMBEDDER_VER` stamp | ✅ | v2 gets a new tag; everything filters by it |
| `ImageEmbedder` (Layer 5) | ✅ | v2 = same class, loads a LoRA adapter |
| `embed_images.py` re-embed | ✅ | reused verbatim to embed the catalog as v2 |
| `fluke_embeddings` (versioned) | ✅ | v2 rows added alongside v1 |
| `photo_id_eval_split` (frozen) | ✅ | the **test set** AND the train/test guardrail |
| `eval.py` (local) | ✅ | re-run on v2; the lift prints in the terminal |

**The one genuinely new contract:** a trained LoRA adapter (a small weights file)
plus version-aware loading in `ImageEmbedder` (when the active version is the LoRA
one, load base CLIP + apply the adapter).

## 4. The integrity crux — train/test discipline

This is the part that makes the lift *honest*, and the frozen split is exactly the
guardrail for it.

- **Test set = the 2,486 held-out query sightings** (`photo_id_eval_split`).
- **Train set = every fluke image EXCEPT those query sightings.** The test
  individuals' *other* photos may train the model (this is closed-set re-ID, which
  matches the v1 baseline) — but the specific held-out query photo must never be
  seen in training, or the lift is a lie.
- **Single-photo whales can't train** (no positive pair) — metric learning needs
  ≥2 images per individual, so training naturally uses the multi-photo individuals.
- **Open-set (later):** additionally reserve some *whole* individuals, never seen in
  training, to measure the abstain/"NOVEL" rate — the crown-jewel "doesn't
  hallucinate identity" number. Deferred; the `role` column on the split table
  reserves the seam.

## 5. How it's exercised — 100% local (the design paying off)

No LangSmith. Run `eval.py` (no `--langsmith`) against v1 and v2 and compare the
printed aggregates. `local_runner` + the `Task`/`Evaluator` protocols were built so
the system-under-test never needs a platform — LoRA is the case that proves it.

## 6. Build sequence (components-first)

0. **Parameterize `embedder_ver`** through the embedder/tool/eval so v1 and v2 can
   be evaluated side by side (today it's a hardcoded constant). Small refactor; the
   prerequisite for any before/after comparison.
1. **Pull training images from S3** — the multi-photo individuals, *excluding* the
   held-out query sightings. (Images were moved to S3; pixels are needed to train.)
2. **Training dataset** — a sampler that yields anchor/positive/negative (or
   per-individual batches) from those images, honoring the exclusion.
3. **LoRA fine-tune script** (`src/training/`) — CLIP ViT-B/32 + LoRA adapters +
   metric-learning loss, trained on MPS. Save the adapter.
4. **Re-embed the catalog** as `clip-vitb32-lora-v2` (reuse `embed_images.py` with
   the v2 embedder) → new rows in `fluke_embeddings`.
5. **Re-run `eval.py` locally** on v2 → compare to the v1 baseline. **The lift is
   the deliverable.**
6. *(optional, later)* open-set holdout → the abstain/NOVEL metric.

Steps 0–1 are plumbing; 2–4 are the actual ML; 5 is a re-run of code we already have.

## 7. Open decisions (the forks — recommendations baked in)

- **Training-individual scope** → *Recommend:* multi-photo individuals (e.g. ≥4–5
  photos for enough positives), excluding held-out queries. Keeps training
  tractable on the Mac and is plenty to demonstrate lift. (Eval stays the existing
  2,486 ≥10-photo split.)
- **Loss function** → *Recommend:* **batch-hard triplet** to start (simple, no fixed
  class head, open-set friendly); note **ArcFace** as the leaderboard-grade upgrade
  (adds an N-individual classification head).
- **LoRA library / base model** → *the main implementation fork.* `peft` integrates
  cleanly with **HuggingFace `transformers` CLIP**; `open_clip` (what v1 uses) needs
  manual LoRA injection. *Recommend:* assess the HF-CLIP + `peft` path first — base
  weights are the **same OpenAI ViT-B/32 checkpoint**, so the v1↔v2 comparison stays
  fair (minor preprocessing differences to confirm). Fallback: inject LoRA into
  `open_clip` attention layers by hand.
- **Success criterion** → any substantial lift over 1.6% / 4.0% proves the technique;
  matching a Kaggle leaderboard is explicitly **not** the goal (portfolio honesty).

## 8. Bounded-scope guardrails (so it stays "garnish, not an ML project")

Off-the-shelf loss, off-the-shelf LoRA, a tractable image subset, measured against
an eval that already exists. One tool's internals only; everything else stays
pretrained. If it starts growing a training framework, stop.
