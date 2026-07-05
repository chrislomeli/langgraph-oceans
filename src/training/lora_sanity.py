"""Part 0 — LoRA wiring sanity check  (FILL-IN-THE-BLANKS).

You wrap the CLIP vision tower with LoRA adapters and prove three things before we
train anything real:
  1. only a TINY fraction of params are trainable (the adapters, ~0.7%)
  2. the base CLIP weights stay FROZEN
  3. a gradient actually reaches the adapters after one backward pass

Run:   uv run python -m training.lora_sanity
Pass:  the ✅ line at the very end.

(The 'UNEXPECTED text_model...' notes on first load are benign — we load only the
vision tower from the full CLIP checkpoint.)
"""

from __future__ import annotations

import torch
from transformers import CLIPVisionModelWithProjection
from peft import LoraConfig, get_peft_model

from core.config import get_settings

# Same OpenAI ViT-B/32 weights as the v1 baseline; outputs a 512-d image embedding
# (matches the fluke_embeddings vector(512) column), so v1↔v2 stays comparable.
HF_MODEL = "openai/clip-vit-base-patch32"
# HF_MODEL = "imageomics/bioclip"

def load_base():
    """The CLIP IMAGE tower + projection head → 512-d embeddings (what we fine-tune)."""
    return CLIPVisionModelWithProjection.from_pretrained(HF_MODEL)


def show_linear_targets(model) -> None:
    """Print the distinct Linear layer names — your LoRA target candidates."""
    names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)})
    print("Linear layers you could target with LoRA:", names)


def wrap_with_lora(model):
    """TODO(you): return `model` wrapped with LoRA adapters using peft.

    The LoRA-meaningful code — this is the part that teaches the technique:
      1.  from peft import LoraConfig, get_peft_model
      2.  build a LoraConfig with:
            r            = the rank (start at 8) — the 'low' in low-rank
            lora_alpha   = scaling (start at 16)
            target_modules = the ATTENTION projection names from show_linear_targets()
                             — you want the q/k/v/output projections, NOT fc1/fc2 or
                             visual_projection
            lora_dropout = 0.05
            bias         = "none"
      3.  return get_peft_model(model, config)

    Hint: peft matches target_modules by the layer's *suffix* name, so a short list
    of names like ["q_proj", ...] applies LoRA to that layer in every block.
    """


    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    return get_peft_model(model, config)


def main() -> None:
    get_settings().apply_hf()
    model = load_base()
    show_linear_targets(model)

    model = wrap_with_lora(model)  # ← your code lives in wrap_with_lora()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)")

    # one dummy image batch → embeddings → a scalar loss → backward
    model.train()
    out = model(pixel_values=torch.randn(2, 3, 224, 224)).image_embeds  # [2, 512]
    out.pow(2).mean().backward()

    lora_grad = any("lora_" in n and p.grad is not None for n, p in model.named_parameters())
    base_frozen = all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
    print(f"embeddings shape: {tuple(out.shape)}")
    print(f"gradient reached LoRA adapters? {lora_grad}")
    print(f"base CLIP frozen?               {base_frozen}")

    ok = lora_grad and base_frozen and (trainable / total < 0.05)
    print("\n✅ Part 0 checkpoint PASSED — LoRA is wired correctly" if ok
          else "\n⚠️  not yet — re-check the TODO against the three conditions above")


if __name__ == "__main__":
    main()
