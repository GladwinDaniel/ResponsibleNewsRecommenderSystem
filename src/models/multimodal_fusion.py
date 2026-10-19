"""
multimodal_fusion.py
--------------------
Multimodal Fusion Module uniting Text Encoders (MuRIL / RoBERTa)
with Vision Backbones (EfficientNet / Xception) for Responsible News Recommendation.

Features:
1. Gated Multimodal Unit (GMU):
   Dynamically gates between textual semantics and visual features.
2. Cross-Modal Attention Fusion:
   Bidirectional attention allowing text tokens to condition on visual patches.
3. Multimodal News Representation:
   Feeds directly into the Heterogeneous Graph Neural Network and Responsible Reranker.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit (GMU).
    Learns an element-wise or dimensional gating vector:
      h_t = tanh(W_t * e_text + b_t)
      h_v = tanh(W_v * e_vis + b_v)
      z   = sigmoid(W_z * [e_text; e_vis] + b_z)
      h   = z * h_t + (1 - z) * h_v
    """

    def __init__(self, text_dim: int = 128, vis_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, out_dim)
        self.vis_proj = nn.Linear(vis_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(text_dim + vis_dim, out_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, text_emb: torch.Tensor, vis_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_emb: (B, text_dim)
            vis_emb: (B, vis_dim)
        Returns:
            fused: (B, out_dim) fused normalized multimodal embedding
            gate_val: (B, out_dim) gating weights (higher = text dominance, lower = vision dominance)
        """
        h_t = torch.tanh(self.text_proj(text_emb))
        h_v = torch.tanh(self.vis_proj(vis_emb))
        z = self.gate(torch.cat([text_emb, vis_emb], dim=-1))
        fused = z * h_t + (1.0 - z) * h_v
        fused = self.layer_norm(fused)
        fused = F.normalize(fused, p=2, dim=-1)
        return fused, z


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module where text queries vision and vision queries text.
    """

    def __init__(self, emb_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha_t2v = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha_v2t = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, text_emb: torch.Tensor, vis_emb: torch.Tensor) -> torch.Tensor:
        # text_emb: (B, D) -> (B, 1, D)
        t_seq = text_emb.unsqueeze(1)
        v_seq = vis_emb.unsqueeze(1)

        # Cross attention
        attn_t, _ = self.mha_t2v(query=t_seq, key=v_seq, value=v_seq)
        attn_v, _ = self.mha_v2t(query=v_seq, key=t_seq, value=t_seq)

        t_out = self.norm1(t_seq + attn_t).squeeze(1)
        v_out = self.norm2(v_seq + attn_v).squeeze(1)

        fused = self.fc(torch.cat([t_out, v_out], dim=-1))
        return F.normalize(fused, p=2, dim=-1)


class MultimodalNewsModel(nn.Module):
    """
    Complete Multimodal News Model unifying:
    - Text: MuRIL or RoBERTa
    - Vision: EfficientNet or Xception
    - Fusion: GMU (Gated Multimodal Unit) or Cross-Attention
    - Prediction Heads: Click probability (Relevance) & Multi-Label Persuasion Risk
    """

    def __init__(
        self,
        text_encoder_name: str = "muril",
        vision_encoder_name: str = "efficientnet",
        fusion_type: str = "gmu",
        emb_dim: int = 128,
        num_persuasion_classes: int = 23,
    ):
        super().__init__()
        self.text_name = text_encoder_name.lower()
        self.vision_name = vision_encoder_name.lower()
        self.fusion_type = fusion_type.lower()
        self.emb_dim = emb_dim

        # Text Encoder
        if "muril" in self.text_name:
            from src.models.muril_encoder import MuRILEncoder
            self.text_encoder = MuRILEncoder(proj_dim=emb_dim, load_pretrained=False)
        else:
            from src.models.persuasion_detector import PersuasionDetector
            self.text_encoder = None  # fallback wrapper

        # Vision Encoder
        from src.models.vision_backbones import NewsVisionExtractor
        self.vision_encoder = NewsVisionExtractor(backbone=self.vision_name, proj_dim=emb_dim)

        # Fusion Unit
        if self.fusion_type == "gmu":
            self.fusion = GatedMultimodalUnit(text_dim=emb_dim, vis_dim=emb_dim, out_dim=emb_dim)
        else:
            self.fusion = CrossModalAttention(emb_dim=emb_dim)

        # Click Prediction Head (Relevance score for AUC)
        self.click_head = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Multi-label Persuasion Risk Head
        self.risk_head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_persuasion_classes),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        vis_emb: torch.Tensor,
        user_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_emb: (B, emb_dim)
            vis_emb: (B, emb_dim)
            user_emb: optional (B, emb_dim)
        Returns:
            Dict containing:
              - 'multimodal_emb': fused article representation
              - 'persuasion_logits': (B, 23)
              - 'click_score': (B, 1) relevance probability if user_emb provided
        """
        if isinstance(self.fusion, GatedMultimodalUnit):
            fused, gate = self.fusion(text_emb, vis_emb)
        else:
            fused = self.fusion(text_emb, vis_emb)
            gate = None

        logits = self.risk_head(fused)
        out = {
            "multimodal_emb": fused,
            "persuasion_logits": logits,
            "gate": gate,
        }

        if user_emb is not None:
            # Dual interaction for click prediction
            pair = torch.cat([user_emb, fused], dim=-1)
            out["click_prob"] = self.click_head(pair).squeeze(-1)

        return out

    def compute_similarity(self, user_emb: torch.Tensor, article_emb: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity relevance between user and multimodal article embeddings."""
        return (F.cosine_similarity(user_emb, article_emb, dim=-1) + 1.0) / 2.0
