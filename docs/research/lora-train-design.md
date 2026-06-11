# Design note — `training/lora_train.py` (Phase-C)

Mid-level design for the real LoRA fine-tune that follows the Part-0 wiring check.
Goal: fine-tune the CLIP fluke embedder so same-whale photos cluster tighter and
different whales spread apart, then measure the lift on the by-individual eval.

This note pins down the decisions and interfaces. It is not the code; it is the
contract the code must satisfy.

---

## 1. What it does, in one paragraph

`lora_train.py` reads the **`train`** whales from `datasets.reid_split`, loads their
JPEGs from `Settings.image_root`, runs them through a **CLIP vision tower wrapped in
LoRA adapters** (base frozen), and minimizes a **metric-learning loss** that pulls
same-individual embeddings together and pushes different ones apart. Every epoch it
checks **Recall@1 on the `val` whales** (disjoint individuals) and keeps the best
adapter checkpoint. The output is a saved adapter that gets registered as a new
`embedder_ver` so `embed_images.py` can re-embed the catalog and the eval can score
the lift.

It reads **images, never the `fluke_embeddings` table** — those vectors are from the
old model and are useless for training the new one.

---

## 2. Decisions (locked in this session)

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Base model, round 1 | OpenAI CLIP ViT-B/32 | Isolate the LoRA variable vs the existing baseline |
| 2 | Holdout | by-individual `reid_split` (train≥2, val/test≥4) | Honest generalization test (unseen whales) |
| 3 | Adapters | LoRA on attention `q/k/v/out_proj`, r=8, base frozen | Proven in `lora_sanity.py` (0.67% trainable, grads flow) |
| 4 | Batch sampling | PK sampler — P identities × K=4 photos | Guarantees positives per batch; draws from the ≥4 depth |
| 5 | **Framework** | **transformers + peft** (NOT open_clip) | See §3 — open_clip's fused attention blocks peft LoRA |
| 6 | Loss, round 1 | batch-hard triplet (PK-friendly, transparent) | See §6 — ArcFace is the round-2 lever |

---

## 3. The framework decision (this revises an earlier suggestion)

Earlier I leaned "train in open_clip so adapters drop into the embedder." That's
**wrong**, and here is why:

- `peft` LoRA targets `nn.Linear` modules by name. transformers' CLIP exposes
  separate `q_proj / k_proj / v_proj / out_proj` Linears → peft wraps them cleanly.
  `lora_sanity.py` already proved this end to end.
- open_clip's ViT uses `nn.MultiheadAttention` with a **fused `in_proj_weight`**
  (a Parameter, not three Linears). peft has nothing to grab. You'd have to hand-roll
  LoRA into a fused projection — custom, fragile, untested.

**So the LoRA lineage lives in transformers+peft.** The cost: the embedder currently
only knows open_clip, so it must learn a transformers backend (§5). The benefit: the
proven path, and the same model class for train and inference (no weight-porting).

**Consequence — a clean baseline.** The existing `clip-vitb32-v1` vectors are
open_clip. To attribute the lift purely to LoRA (not to open_clip-vs-transformers
implementation drift), establish the comparison baseline with the **same transformers
loader, adapters off**:

- `clip-hf-vitb32-v1` — transformers CLIP, no adapters → the honest baseline.
- `clip-hf-vitb32-lora-v1` — transformers CLIP + trained adapters.

Compare those two on the `test` whales. (Pragmatic shortcut: open_clip and
transformers load the *same* OpenAI weights, so `clip-vitb32-v1` is a *near* baseline
— fine for a first look, but the same-loader pair is the rigorous version.)

---

## 4. Data flow

```
reid_split (split='train')        Settings.image_root/<ind>/<sighting>.jpg
        │                                   │
        └──────────► PK sampler ◄───────────┘
                         │  batch = P whales × K=4 photos (image tensors + labels)
                         ▼
        CLIP vision tower + LoRA adapters (base frozen)   ← transformers + peft
                         │  L2-normalized 512-d embeddings
                         ▼
              metric loss (batch-hard triplet)
                         │  backward → AdamW updates ADAPTERS ONLY
                         ▼
   every epoch: embed val whales in-memory → leave-one-out Recall@1 → keep best
                         │
                         ▼
        save adapter checkpoint → artifacts/lora/clip-hf-vitb32-lora-v1/
```

Validation is **in-memory**: embed the val images with the current model, do the
same leave-one-out retrieval among val whales, compute Recall@1. No DB writes during
training — the table only gets touched once, after, by `embed_images.py`.

---

## 5. Interfaces this changes elsewhere

### 5a. `EmbedderSpec` gains a backend + adapter path
```python
@dataclass(frozen=True)
class EmbedderSpec:
    model_name: str
    pretrained: str | None
    dim: int
    backend: str = "open_clip"        # "open_clip" | "hf"
    adapter_path: str | None = None   # peft checkpoint dir, for LoRA vers
```
Registry additions:
```python
"clip-hf-vitb32-v1":      EmbedderSpec("openai/clip-vit-base-patch32", None, 512, backend="hf"),
"clip-hf-vitb32-lora-v1": EmbedderSpec("openai/clip-vit-base-patch32", None, 512,
                                       backend="hf", adapter_path="artifacts/lora/clip-hf-vitb32-lora-v1"),
```

### 5b. `ImageEmbedder.load_model` branches on backend
- `open_clip`: today's path (baseline, bioCLIP).
- `hf`: `CLIPVisionModelWithProjection.from_pretrained(model_name)`; if `adapter_path`,
  wrap with `PeftModel.from_pretrained(base, adapter_path)`; encode via
  `model(pixel_values=...).image_embeds`; preprocess with the HF `CLIPImageProcessor`.
- Both paths still **L2-normalize** so cosine space is identical, and the §dim guard
  still applies.

> The image preprocessing MUST match between training and embedding (same
> `CLIPImageProcessor` config). A train/inference preprocessing mismatch silently
> degrades every vector — treat it as part of the model identity.

### 5c. Nothing else changes
`embed_images.py`, `photo_id` tool, and the eval already key off `embedder_ver`. Once
the LoRA ver is registered and the catalog re-embedded, they Just Work.

---

## 6. The loss (round 1: batch-hard triplet)

With a PK batch (P whales × K photos), for each anchor pick the **hardest positive**
(same whale, farthest) and **hardest negative** (different whale, nearest):

```
L = mean over anchors of  max(0,  margin + d(a, hardest_pos) − d(a, hardest_neg))
```

- Transparent: it *is* the pull-together / push-apart intuition we built, made literal.
- PK-friendly: the batch structure guarantees ≥1 positive and many negatives per anchor.
- No extra parameters beyond the adapters.

**Documented alternatives (round-2 levers, not round-1):**
- **Supervised contrastive (SupCon)** — uses *all* positives in the batch, often beats
  triplet, still PK-native. Low-risk upgrade.
- **ArcFace / margin-softmax** — treat each train whale as a class with an angular
  margin; SOTA-ish for re-ID, very stable, no mining. Cost: a proxy head of
  `512 × 11,789` (~6M params, training-only, discarded at inference) and it's not
  low-rank. Strong candidate once the pipeline works.

---

## 7. Training config (starting points, tune on val)
- Trainable: LoRA adapters only (base frozen — assert via the Part-0 checks).
- Optimizer: AdamW; LR ~3e-4 for adapters (LoRA tolerates higher LR than full FT);
  cosine schedule with warmup.
- Batch: P=32 whales × K=4 = 128 images; a few epochs (early-stop on val Recall@1).
- `lora_alpha`: set to `2*r` to keep `alpha/r` scaling at 2.0 (we noticed pinning
  alpha=16 at r=16 silently halved it).
- Optional round-2 knob: also adapt `visual_projection` (the 512-d head) — the most
  task-specific layer. Left out of Part-0 deliberately.

---

## 8. Outputs / artifacts
- `artifacts/lora/clip-hf-vitb32-lora-v1/` — peft adapter checkpoint (`save_pretrained`).
- A training log / val-Recall@1 curve (sanity that it learned).
- Registry entry + (optional) the `clip-hf-vitb32-v1` baseline vectors.

`artifacts/` should be gitignored (like the data); the registry entry + this note are
the durable record.

---

## 9. Risks / watch-list
- **Framework drift** (§3) — train and embed must both be transformers+peft. The
  same-loader baseline removes the open_clip confound.
- **Preprocessing mismatch** (§5b) — silent vector degradation.
- **Trivial positives** — near-duplicate same-encounter photos give zero gradient;
  batch-hard mining mostly dodges this, but worth spot-checking.
- **Overfitting to train identities** — the whole reason val/test are disjoint
  individuals; if val Recall@1 diverges from train loss, that's the tell.
- **Re-embed cost** — one full catalog pass (~114k images) per trained model. Budget it.

---

## 10. Build order (tasks)
1. Add `backend`/`adapter_path` to `EmbedderSpec`; teach `ImageEmbedder` the `hf` path
   (no adapters yet). Register + embed `clip-hf-vitb32-v1`; eval it → the clean baseline.
2. `reid_split` loader: train image/label iterator + PK sampler.
3. `lora_train.py`: wrap (reuse `lora_sanity.wrap_with_lora`), triplet loss, AdamW loop,
   in-memory val Recall@1, checkpoint best.
4. Register `clip-hf-vitb32-lora-v1` with the adapter path; `embed_images.py` re-embed.
5. Run the by-individual eval on `test`; compare Recall@1 vs the §1 baseline → the lift.

Step 1 is independent and de-risks the whole thing (proves the hf backend + clean
baseline before any training). Do it first.
