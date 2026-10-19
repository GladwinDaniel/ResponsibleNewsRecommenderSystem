"""
muril_encoder.py
----------------
MuRIL (Multilingual Representations for Indian Languages) text encoder
for Responsible News Recommendation.

Backbone: google/muril-base-cased
- Pre-trained on 17 Indian languages and Indian English by Google Research
- Handles multilingual headlines, transliteration, code-switching, and cultural nuances
- Produces dense 768-dim contextual text embeddings
- Multi-label classification head for 23 SemEval-2023 persuasion techniques
- Projection head to align with GNN and multimodal visual spaces
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

TECHNIQUES = [
    "Appeal_to_Authority", "Appeal_to_Fear-Prejudice", "Bandwagon",
    "Black-and-White_Fallacy", "Causal_Oversimplification", "Doubt",
    "Exaggeration-Minimisation", "Flag-Waving", "Glittering_generalities",
    "Loaded_Language", "Name_Calling-Labeling", "Obfuscation",
    "Reasoning", "Red_Herring", "Repetition", "Slogans",
    "Thought-terminating_Clichés", "Transfer", "Whataboutism",
    "Straw_Men", "Appeal_to_Hypocrisy", "Consequential_Oversimplification",
    "Appeal_to_Time",
]
NUM_LABELS = len(TECHNIQUES)


class MuRILEncoder(nn.Module):
    """
    MuRIL (google/muril-base-cased) transformer encoder.
    Outputs:
      1. article_embedding: normalized dense vector (e.g. 128-d or 256-d) for GNN / relevance
      2. persuasion_logits: 23-d multi-label logits for rhetorical technique detection
    """

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        num_labels: int = NUM_LABELS,
        proj_dim: int = 128,
        dropout: float = 0.15,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.proj_dim = proj_dim
        self.encoder = None

        if load_pretrained:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.encoder = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.hidden = self.encoder.config.hidden_size  # 768
                log.info("Successfully loaded MuRIL backbone: %s", model_name)
            except Exception as e:
                log.warning("MuRIL from HuggingFace failed (%s). Initializing native multilingual fallback.", e)
                self.encoder = None

        if self.encoder is None:
            self.hidden = 768
            # Native vocabulary embedding fallback
            self._vocab_emb = nn.Embedding(50_000, self.hidden)
            self._transformer_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden, nhead=8, dim_feedforward=1024, dropout=dropout, batch_first=True
            )
            self._transformer = nn.TransformerEncoder(self._transformer_layer, num_layers=2)
            self.tokenizer = None

        self.dropout = nn.Dropout(dropout)

        # Multi-label Persuasion Detection Head (23 rhetorical classes)
        self.persuasion_head = nn.Sequential(
            nn.Linear(self.hidden, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_labels),
        )

        # Article Embedding Projection Head (for recommendation similarity & GNN)
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, proj_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            text_emb: (B, proj_dim) normalized semantic article embedding
            persuasion_logits: (B, num_labels) logits for 23 persuasion techniques
        """
        if self.encoder is not None:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Use mean pooling with attention mask for richer representation than [CLS]
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(out.last_hidden_state).float()
                sum_embeddings = torch.sum(out.last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = out.last_hidden_state[:, 0, :]
        else:
            emb = self._vocab_emb(input_ids)
            pooled = self._transformer(emb).mean(dim=1)

        pooled = self.dropout(pooled)
        persuasion_logits = self.persuasion_head(pooled)
        text_emb = self.proj_head(pooled)
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        return text_emb, persuasion_logits

    def encode_texts(self, texts: list[str], max_len: int = 128, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method to encode raw string list into embeddings and persuasion probabilities."""
        self.eval()
        if self.tokenizer is not None:
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                emb, logits = self(enc["input_ids"], enc["attention_mask"])
                probs = torch.sigmoid(logits)
                return emb, probs
        else:
            # Hash-based deterministic token ids for offline fallback
            batch_ids = []
            for t in texts:
                ids = [(hash(w) % 49999) + 1 for w in t.split()[:max_len]]
                if not ids:
                    ids = [0]
                batch_ids.append(ids)
            max_l = max(len(ids) for ids in batch_ids)
            padded = [ids + [0] * (max_l - len(ids)) for ids in batch_ids]
            inp = torch.tensor(padded, dtype=torch.long, device=device)
            with torch.no_grad():
                emb, logits = self(inp)
                probs = torch.sigmoid(logits)
                return emb, probs
