# LoRA fine-tune — a fill-in-the-blanks tutorial (you execute, I scaffold)

> **Goal:** learn LoRA by *doing* it — fine-tune the fluke embedder, measure the lift
> over the raw-CLIP baseline (`reid@1=1.6%`), entirely **locally** (no LangSmith).
>
> **How this works:** I provide the plumbing; you write the LoRA-meaningful lines
> (marked `TODO(you)`). Every part ends in a **checkpoint** — a concrete "you should
> see X" — so you're never guessing whether it worked. When a checkpoint passes, ping
> me and I'll deliver the next part. Stuck on a wall? Ping me — this isn't abandonment.

Design rationale lives in `lora-design-note.md`. This file is the hands-on path.

---

## The map (viability-first)

| Part | You build | Checkpoint | Answers |
|---|---|---|---|
| **0** | wrap CLIP in LoRA | adapters trainable, base frozen, gradient flows | *is the wiring right?* |
| 1 | pull ~20 whales' images from S3, load a batch | `(image, individual_id)` batches load | data works |
| 2 | the triplet loss; overfit the tiny set | train loss drops; same-whale cosine rises | **machinery learns** |
| 3 | re-embed the tiny set, run our `eval.py` | `reid@1` vs baseline on the tiny set | **viability!** |
| 4 | scale up, re-embed catalog as v2, full eval | the real lift, v2 vs v1 | the payoff |

Parts 1–3 are a cheap **viability spike**: if LoRA doesn't move the needle on 20 whales
in an afternoon, we learn that fast and rethink — before building anything big.

---

## Part 0 — wire CLIP for LoRA  ✅ ready now

**Concept (just enough):** LoRA freezes the big pretrained model and injects a pair of
small low-rank matrices (`A·B`, rank `r`) next to chosen layers. Only those tiny
matrices train — so you adapt CLIP to flukes by learning **~0.7%** of the parameters
instead of all 88M. `peft` does the injection; you tell it *which* layers (the
attention projections) and *how big* (`r`).

**Why HuggingFace CLIP (not our open_clip):** `peft` targets `nn.Linear` layers, and HF
CLIP exposes clean `q_proj/k_proj/v_proj/out_proj`. open_clip fuses them into a
`MultiheadAttention` peft can't wrap. HF CLIP uses the *same OpenAI ViT-B/32 weights*
and outputs 512-d, so the comparison stays fair (we'll embed an HF-base control in
Part 3).

**Your task:** open `src/training/lora_sanity.py` and implement `wrap_with_lora()` — the
one `TODO(you)`. Build a `peft.LoraConfig` (pick `r`, `lora_alpha`, `target_modules`,
`lora_dropout`, `bias`) and `get_peft_model(model, config)`. The script already prints
your target candidates when you run it.

**Run:**
```
uv run python -m training.lora_sanity
```

**Checkpoint — you should see:**
```
trainable: 589,824 / 88,439,040  (0.67%)      # tiny — the LoRA promise
embeddings shape: (2, 512)
gradient reached LoRA adapters? True
base CLIP frozen?               True
✅ Part 0 checkpoint PASSED — LoRA is wired correctly
```
(The `UNEXPECTED text_model...` lines on first load are benign — we load the vision
tower only.)

**Things to try once it passes (this is the learning):** change `r` (4 vs 8 vs 16) and
watch the trainable-param count move; target only `["q_proj","v_proj"]` vs all four and
see the count change. You're feeling what the knobs do.

→ **When you see the ✅, ping me and I'll deliver Part 1.**

---

## Parts 1–4 — delivered as you go

Stubbed on purpose — each arrives when you clear the prior checkpoint, so the tutorial
stays paced to your execution, not dumped all at once.

- **Part 1 — tiny data:** pull a handful of well-sampled whales' images from S3
  (excluding the held-out query sightings — the train/test guardrail), into a simple
  `(pixels, individual_id)` loader.
- **Part 2 — loss + overfit:** you write the batch-hard **triplet loss**; train on the
  tiny set and watch it memorize (proves the loop learns).
- **Part 3 — viability eval:** re-embed the tiny set with base-HF and with LoRA, run the
  existing `eval.py` on both → does `reid@1` move?
- **Part 4 — the real lift:** scale individuals + steps, re-embed the catalog as
  `clip-vitb32-lora-v2`, full local eval vs the v1 baseline.
