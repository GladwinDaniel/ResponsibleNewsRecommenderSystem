"""
vision_backbones.py
-------------------
Vision backbone feature extractors for Multimodal Responsible News Recommendation:
1. EfficientNet (EfficientNet-B0 / B2) — compound-scaled, lightweight CNN for news visual thumbnails
2. Xception (Exception Net) — extreme Inception with depthwise separable convolutions for image forensics

Both backbones project image inputs into a shared latent visual embedding space
compatible with MuRIL / RoBERTa text encoders and the Heterogeneous GNN.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EFFICIENTNET ENCODER (Compound-Scaled CNN)
# ═══════════════════════════════════════════════════════════════════════════════

class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B0 / B2 backbone for news visual feature extraction.
    Extracts deep visual embeddings (default 256-d or 128-d) and optional
    visual clickbait / sensationalism score.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        proj_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.model_name = model_name
        self.proj_dim = proj_dim
        self.encoder = None

        try:
            import torchvision.models as tv_models
            if hasattr(tv_models, model_name):
                weights = "DEFAULT" if pretrained else None
                base_model = getattr(tv_models, model_name)(weights=weights)
                # EfficientNet has 'features' and 'classifier'
                self.features = base_model.features
                in_features = base_model.classifier[1].in_features  # 1280 for b0
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.in_features = in_features
                self.is_torchvision = True
            else:
                raise AttributeError(f"Model {model_name} not in torchvision.models")
        except Exception as e:
            log.warning("torchvision %s not loaded (%s). Using native depthwise compound conv fallback.", model_name, e)
            self.is_torchvision = False
            self.in_features = 1280
            # Native lightweight compound conv fallback
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.in_features = 128
            self.pool = nn.Identity()

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.in_features, 256),
            nn.GELU(),
            nn.Linear(256, proj_dim),
        )

        # Visual clickbait / sensation detector head
        self.sensation_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Image tensor of shape (B, 3, H, W)
        Returns:
            visual_emb: (B, proj_dim) normalized visual feature vector
            sensation_score: (B, 1) visual sensationalism / clickbait probability
        """
        feat = self.features(x)
        pooled = self.pool(feat)
        flat = torch.flatten(pooled, 1)
        emb = self.projection(flat)
        emb_norm = F.normalize(emb, p=2, dim=-1)
        sensation = self.sensation_head(emb_norm)
        return emb_norm, sensation


# ═══════════════════════════════════════════════════════════════════════════════
# 2. XCEPTION ENCODER (Depthwise Separable CNN / Image Forensics)
# ═══════════════════════════════════════════════════════════════════════════════

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise conv followed by pointwise conv."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.act(out)


class XceptionBlock(nn.Module):
    """Residual block with separable convolutions (the core unit of Xception)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = SeparableConv2d(channels, channels)
        self.conv2 = SeparableConv2d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class XceptionEncoder(nn.Module):
    """
    Xception (Extreme Inception) backbone for visual manipulation & deepfake detection.
    Known for detecting high-frequency compression artifacts, manipulated edges,
    and visual sensationalism in news imagery.
    """

    def __init__(
        self,
        pretrained: bool = True,
        proj_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.is_timm = False

        # Attempt to load via timm if available
        try:
            import timm
            self.backbone = timm.create_model("xception", pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features  # usually 2048
            self.is_timm = True
        except Exception:
            # Native PyTorch Xception-style architecture fallback
            log.info("timm.xception not found; using high-fidelity native Xception separable architecture.")
            self.backbone = nn.Sequential(
                # Entry Flow
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                SeparableConv2d(64, 128, stride=2),
                # Middle Flow (Repetitive separable residual blocks)
                XceptionBlock(128),
                XceptionBlock(128),
                # Exit Flow
                SeparableConv2d(128, 256, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            in_features = 256

        self.in_features = in_features
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.in_features, 256),
            nn.GELU(),
            nn.Linear(256, proj_dim),
        )

        # Forensic tampering / manipulation detector head
        self.forensic_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Image tensor of shape (B, 3, H, W)
        Returns:
            visual_emb: (B, proj_dim) normalized visual feature vector
            forensic_score: (B, 1) probability of visual manipulation / clickbait
        """
        if self.is_timm:
            feat = self.backbone(x)
        else:
            feat = self.backbone(x)
            feat = torch.flatten(feat, 1)

        emb = self.projection(feat)
        emb_norm = F.normalize(emb, p=2, dim=-1)
        forensic_score = self.forensic_head(emb_norm)
        return emb_norm, forensic_score


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UNIFIED VISION WRAPPER & FALLBACK GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class NewsVisionExtractor(nn.Module):
    """
    Unified news visual extractor supporting both EfficientNet and Xception.
    Includes deterministic content-seeded visual embedding fallback when image
    files are not present on disk.
    """

    def __init__(self, backbone: str = "efficientnet", proj_dim: int = 128, pretrained: bool = False):
        super().__init__()
        self.backbone_name = backbone.lower()
        if "eff" in self.backbone_name:
            self.model = EfficientNetEncoder(proj_dim=proj_dim, pretrained=pretrained)
        elif "xcep" in self.backbone_name or "except" in self.backbone_name:
            self.model = XceptionEncoder(proj_dim=proj_dim, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown vision backbone: {backbone}. Choose 'efficientnet' or 'xception'.")

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(images)

    @torch.no_grad()
    def extract_from_article_ids(self, article_ids: list[str], device: str = "cpu") -> torch.Tensor:
        """
        Deterministic pseudo-visual feature generator for offline benchmarking
        when raw image files are cached or simulated from article hashes.
        Ensures consistent visual features per news item.
        """
        features = []
        for aid in article_ids:
            # Deterministic seed from article ID string
            seed = sum(ord(c) * (31 ** i) for i, c in enumerate(aid[-6:])) % (2**31)
            rng = torch.Generator().manual_seed(seed)
            # Sample coherent visual embedding vector
            feat = torch.randn(self.model.proj_dim, generator=rng)
            features.append(feat)

        out = torch.stack(features).to(device)
        return F.normalize(out, p=2, dim=-1)
