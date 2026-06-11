"""embed_raw.py — the HF image→vector pipeline, UNROLLED (no ImageEmbedder wrapper).

Every line here is exactly what `_load_hf` + `_encode` do inside ImageEmbedder, just
with the abstraction removed so each tensor is visible. Throwaway learning scaffold.
(This is PURE INFERENCE — no LoRA. For the LoRA training toy see `lora_raw.py`.)

THE WHOLE FLOW, AS SHAPES (watch the shape change — that's the story):
    PIL image (any H×W)
      └─ processor      → [1,3,224,224] → squeeze → [3,224,224]   model-ready pixels
      └─ torch.stack    → [1,3,224,224]                           a batch (N=1)
      └─ .to(device)    → [1,3,224,224] on GPU/MPS                data meets model
      └─ model(...).image_embeds → [1,512]                        ← THE EMBEDDING
      └─ / norm         → [1,512] unit-length                     cosine-ready
      └─ .cpu().tolist()→ [[512 floats]]                          torch → python
      └─ [0]            → [512 floats]                             what gets stored

A TENSOR is just a multi-dimensional array of numbers PyTorch can run on a GPU.
Its SHAPE lists the size of each dimension:
    [3,224,224] = one image: 3 color channels (R,G,B), each a 224×224 pixel grid.
    [1,512]     = 1 row of 512 numbers.

    uv run python -m training.embed_raw
"""

import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image

from config import get_settings
from models.image_embedder import pick_device, resolve_spec

# get the basic hf model and device
model_name = "clip-hf-vitb32-v1"
spec = resolve_spec(model_name)
device = pick_device()
if spec.backend != "hf":
    raise Exception("just testing hf for now")

# load model (the IMAGE tower + 512-d projection head; eval() = inference mode)
model = CLIPVisionModelWithProjection.from_pretrained(spec.model_name).to(device).eval()
processor = CLIPImageProcessor.from_pretrained(spec.model_name)

# get an image to process
print("model class :", type(model).__name__)
print("backend     :", spec.backend)
images = sorted(get_settings().image_root.glob("*/*.jpg"))
single_image = images[0]

# open PIL — load the JPEG off disk; convert('RGB') guarantees 3 channels (drops alpha,
# upgrades greyscale), so the tensor is always [3, H, W].
img = Image.open(single_image).convert("RGB")

# --- THE PROCESSOR: turn a PIL image of ANY size into the EXACT tensor CLIP expects -----
# It (1) resizes short-side to 224 + center-crops to 224×224 (CLIP's fixed input size),
#    (2) converts pixels to a float tensor scaled ~[0,1],
#    (3) normalizes each channel with CLIP's mean/std.
# Step 3 is correctness-critical: the model was TRAINED on inputs normalized exactly this
# way — wrong processor → every vector silently degrades (preprocessing = model identity).
#   return_tensors="pt" → give me PyTorch tensors (not numpy).
#   .pixel_values       → the image tensor field, shape [1, 3, 224, 224]
#                         (the processor ALWAYS adds a leading batch dim — that 1).
#   .squeeze(0)         → drop dim-0 because it's size 1 → [3, 224, 224]  (one image).
tensor = processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

# --- BUILD A BATCH: models process MANY images at once, shape [N, 3, 224, 224] -----------
# torch.stack takes a LIST of equal-shaped tensors and joins them along a NEW leading dim.
#   [ [3,224,224] ]  → [1,3,224,224]   (N=1 here; ten images would give [10,3,224,224]).
# (Note: we squeezed the batch dim off above, then re-add it here — the real _encode gets a
#  list of per-image [3,224,224] tensors and stacks ALL of them; for one image it round-trips.)
#   .to(device) → move the INPUT onto the same hardware as the model (required, or it errors).
batch = torch.stack([tensor]).to(device)

# --- THE FORWARD PASS: run the image through the neural net → the 512-d embedding ---------
# torch.no_grad(): we're not training, so don't track gradients → faster, less memory.
#
# WHAT HAPPENS INSIDE model(pixel_values=batch)  — CLIP ViT-B/32 ("B"=base, "32"=patch size):
#
#   1. PATCHIFY. The 224×224 image is cut into non-overlapping 32×32 squares →
#      (224/32)² = 7×7 = 49 patches. Each patch (3×32×32 = 3072 pixel numbers) is linearly
#      projected into a 768-d vector ("patch embedding"). So the image becomes 49 vectors —
#      think of it as turning the picture into a sentence of 49 "visual words."
#
#   2. ADD A [CLS] TOKEN + POSITIONS. A special learnable [CLS] vector is prepended (→ 50
#      tokens). Positional embeddings are added so the model knows WHERE each patch sat
#      (a transformer is otherwise order-blind).
#
#   3. 12 TRANSFORMER LAYERS, each = SELF-ATTENTION + a small MLP:
#      SELF-ATTENTION is the heart. Every token builds a Query, a Key, and a Value (via the
#      q_proj / k_proj / v_proj linear layers). Each token's Query is compared to every
#      token's Key → attention weights ("how much should I look at you?"), then it pulls a
#      weighted mix of everyone's Values (combined by out_proj). Net effect: each patch can
#      relate to distant patches — e.g. tie a notch on one edge to the trailing-edge shape.
#      ► THESE q/k/v/out_proj LINEARS ARE EXACTLY WHAT LoRA ADAPTS. That's why Part 0 targets
#        them: nudging how the model attends is how we teach it fluke-specific identity.
#
#   4. POOL + PROJECT. After 12 layers, take the [CLS] token's final 768-d vector (its
#      learned summary of the whole image), then visual_projection maps 768 → 512.
#      That 512-d result is `.image_embeds`.
#
# So: pixels → 49 patch-words → attention mixes them over 12 layers → one 768-d summary →
#     projected to 512. The "150,528 pixels → 512 meaning-numbers" compression, concretely.
with torch.no_grad():
    feats = model(pixel_values=batch).image_embeds  # [1, 512]  ← THE EMBEDDING

    # --- L2 NORMALIZE to unit length (yes, this is normalizing) ---------------------------
    # feats.norm(dim=-1) = Euclidean length √(Σ x²) along the LAST dim (the 512 axis) →
    #   one length per image, shape [1]. keepdim=True keeps it [1,1] so it BROADCASTS:
    #   dividing [1,512] by [1,1] divides all 512 numbers by that length → each row length 1.
    # WHY: so cosine similarity == dot product. Cosine cares about DIRECTION, not magnitude;
    #   putting every vector on the unit sphere makes comparisons pure angle. (→ norm prints 1.)
    feats = feats / feats.norm(dim=-1, keepdim=True)  # [1, 512], unit-length

# --- TORCH → PYTHON: cross back from tensor-world to plain floats -------------------------
#   .cpu()    → move the tensor off GPU/MPS back to CPU memory (can't convert a GPU tensor
#               straight to Python; and Postgres/JSON live in CPU/Python land).
#   .tolist() → convert to nested plain Python lists of floats → [[512 floats]]
#               (outer = batch, inner = the 512 numbers).
vectors = feats.cpu().tolist()
vector = vectors[0]  # the 512-float list for our one image — what gets stored in the DB

print("length:", len(vector))
print("norm  :", round(sum(x * x for x in vector) ** 0.5, 6))
print("first3:", [round(x, 4) for x in vector[:3]])
