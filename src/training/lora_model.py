"""lora_model.py — build a CLIP image tower wrapped in (untrained) LoRA adapters.

THE shared 'build' step for everything LoRA. The Part-0 sanity check, the overfit
toy, and the real trainer should all call get_lora_model() so the model is constructed
exactly ONE way.

What it returns is an UNtrained model — the adapter is zeroed, ready to train. Loading
a model that ALREADY has a trained adapter is a different path (PeftModel.from_pretrained,
see image_embedder._load_hf). Two verbs, kept apart:
    get_peft_model(base, config)         → CREATE a fresh adapter to train   (here)
    PeftModel.from_pretrained(base, dir)  → LOAD a saved adapter to use      (inference)
"""

from __future__ import annotations

from collections.abc import Callable

from peft import LoraConfig, get_peft_model
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from models.image_embedder import pick_device, resolve_spec


def get_lora_model(
    model_name: str,
    device: str | None = None,
    r: int = 8,
    lora_alpha: int = 16,
) -> tuple[object, Callable]:
    """Load the HF CLIP image tower for `model_name` and bolt on fresh LoRA adapters.

    Returns (model, preprocess):
      model      — frozen CLIP + UNtrained LoRA adapters (a PeftModel), on `device`.
      preprocess — PIL image -> [3,224,224] tensor, the model's MATCHED processor.

    `r` / `lora_alpha` are exposed so you can wiggle the rank (the Part-0 experiment)
    without editing this file.
    """
    spec = resolve_spec(model_name)
    if spec.backend != "hf":
        raise ValueError(
            f"{model_name!r} is backend={spec.backend!r}; LoRA needs the hf CLIP "
            "(peft can only target HF's q/k/v/out_proj linears)."
        )
    device = device or pick_device()

    # 1. the frozen foundation: the image half of CLIP (+ the 512-d projection head)
    base = CLIPVisionModelWithProjection.from_pretrained(spec.model_name)

    # 2. bolt on the trainable adapters (A·B) at the attention projections
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base, config).to(device)  # foundation frozen, adapter trainable

    # 3. the matched image→tensor converter (model + processor MUST come from the same name)
    processor = CLIPImageProcessor.from_pretrained(spec.model_name)

    def preprocess(image):
        return processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

    return model, preprocess
