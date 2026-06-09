"""training — offline LoRA fine-tuning of the image embedder (Phase C, item 13).

NOT a runtime layer. Produces a LoRA adapter + a new embedder_ver; the catalog is
then re-embedded and the existing Layer-A eval re-run LOCALLY to measure the lift.
Bounded to one tool's internals — everything else stays pretrained.
"""
