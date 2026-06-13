"""reid_model.py — the v3 fluke embedder: EfficientNetV2-S → GeM → 512-d (inference path).

This is the INFERENCE module — backbone + pooling + embedding head, producing one 512-d
vector per image. The ArcFace classification head used during training lives in the trainer
and is discarded here: at inference we only need the embedding, then cosine retrieval (same
as every prior embedder). Output is 512-d on purpose so it reuses fluke_embeddings.vector(512).

ImageEmbedder loads this via the `timm` backend; `_encode` calls forward() and L2-normalizes.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# EffNetV2-S pretrained on ImageNet-21k then fine-tuned on 1k — a strong, compute-tractable
# starting point. Default input 384px (the resolution lever the probe identified).
DEFAULT_BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"


class GeM(nn.Module):
    """Generalized-mean pooling — the re-ID standard; a learnable middle ground between
    average (p=1) and max (p→∞) pooling that emphasizes the most discriminative regions."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [N, C, H, W] → [N, C]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        return x.flatten(1)


class ReIDModel(nn.Module):
    """Backbone → GeM → BN → Linear(→emb_dim) → BN. forward returns the embedding
    (pre-L2-norm; the caller normalizes, matching the other backends' contract)."""

    def __init__(self, backbone: str = DEFAULT_BACKBONE, emb_dim: int = 512, pretrained: bool = True):
        super().__init__()
        # num_classes=0, global_pool="" → forward_features gives the raw [N,C,H,W] map for GeM
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        feat_dim = self.backbone.num_features
        self.pool = GeM()
        self.bn1 = nn.BatchNorm1d(feat_dim)
        self.fc = nn.Linear(feat_dim, emb_dim)
        self.bn2 = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [N, 3, H, W] → [N, emb_dim]
        f = self.backbone.forward_features(x)
        f = self.pool(f)
        f = self.bn1(f)
        f = self.fc(f)
        f = self.bn2(f)
        return f


def build_preprocess(backbone: str = DEFAULT_BACKBONE, image_size: int = 384):
    """timm's own eval transform for this backbone, forced to image_size — PIL → [3,H,W]."""
    cfg = timm.data.resolve_model_data_config({"architecture": backbone})
    cfg["input_size"] = (3, image_size, image_size)
    return timm.data.create_transform(**cfg, is_training=False)
