"""lora_raw.py — the LoRA overfit toy, UNROLLED in ONE self-contained file.

Read this top-to-bottom to see the whole LoRA training loop with nothing hidden:
build model → embed 8 photos of 2 whales → loss → forward/backward/step → watch the
same/different cosines FLIP. Deliberately self-contained (it duplicates the build that
`lora_model.get_lora_model` does) so it stays readable as a learning reference. Real
code uses `lora_model.get_lora_model` instead of inlining the build.

    uv run python -m training.lora_raw
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from core.config import get_settings
from models.image_embedder import pick_device, resolve_spec
from PIL import Image

# --- the tiny dataset: (sighting photo, whale-id label) ----------------------------
# index 0,1 = whale 12 's first two photos (the "12a/12b" from our cosine check)
# index 4,5 = whale 18 's first two photos (the "18a/18b")
PHOTOS = [
    (12, "12/18621.jpg"), (12, "12/22110.jpg"), (12, "12/99277.jpg"), (12, "12/103142.jpg"),
    (18, "18/1789.jpg"),  (18, "18/2667.jpg"),  (18, "18/2891.jpg"),  (18, "18/3761.jpg"),
]

def main():
    s = get_settings()
    s.apply_hf()

    # get the basic hf model and device
    spec = resolve_spec('clip-hf-vitb32-v1')
    device = pick_device()
    if spec.backend != "hf":
        raise Exception("just testing hf for now")


    # LoRA config
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    # Loads the image half of CLIP from HuggingFace. The full CLIP has two towers, image and text.
    # This class is only the image tower (+ the small head that projects to a 512-d vector). We never embed text, so the text tower is left out.
    # So: this is your embedder — photo in, 512 numbers out. Untrained-for-flukes, just the stock OpenAI weights.
    base = CLIPVisionModelWithProjection.from_pretrained(spec.model_name)

    # This takes your base embedder and injects the LoRA adapters into the layers named in config.target_modules (q_proj, k_proj, v_proj, out_proj).
    # It returns a new object that contains the frozen base plus the tiny trainable adapters.
    # After this line, base weights are frozen, only adapters will move.
    model = get_peft_model(base, config)

    # Moves the whole thing (frozen base + adapters) onto your GPU (mps on your Mac).
    model = model.to(device)

    # processor
    processor = CLIPImageProcessor.from_pretrained(spec.model_name)
    def preprocess(image):
        return processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

    # images/labels
    file_values = [preprocess(Image.open(s.image_root / f).convert("RGB")) for _, f in PHOTOS]
    label_values = [y for y, _ in PHOTOS]

    #
    pixels = torch.stack(file_values).to(device)
    labels = torch.tensor(label_values).to(device)

    #
    same = labels[:, None] == labels[None, :]  # True where two photos share a whale
    off_diag = ~torch.eye(len(labels), dtype=torch.bool, device=device)  # ignore self-pairs


    def embed():
        feats = model(pixel_values=pixels).image_embeds
        return feats / feats.norm(dim=-1, keepdim=True)  # unit vectors → dot = cosine


    def report(tag):
        with torch.no_grad():
            v = embed()                  # one forward pass
            cos = v @ v.t()              # 8x8 cosine matrix
            same_mean = cos[same & off_diag].mean().item()
            diff_mean = cos[~same].mean().item()
        print(f"{tag:6}  SAME avg={same_mean:.3f}   DIFFERENT avg={diff_mean:.3f}   "
              f"(want SAME > DIFFERENT)")


    report("BEFORE")

    # 3. the loop we drew: nudge the adapter to shrink same-distance, grow diff-distance
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for step in range(300):
        v = embed()                  # one forward pass
        cos = v @ v.t()
        same_dist = (1 - cos[same & off_diag]).mean()  # same photos: want CLOSE  (small)
        diff_dist = (1 - cos[~same]).mean()  # diff photos: want FAR    (large)
        loss = same_dist - diff_dist  # minimizing pulls same in, pushes diff out
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:3} ::  same={same_dist} diff={diff_dist} loss={loss.item():+.3f}")

    report("AFTER")

if __name__ == "__main__":
    main()