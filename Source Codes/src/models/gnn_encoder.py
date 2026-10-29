"""
gnn_encoder.py
--------------
2-layer GraphSAGE heterogeneous GNN encoder.

Architecture
------------
  Input  : HeteroData with user / article / topic / entity node features
  Layers : 2 × HeteroConv(SAGEConv) → ReLU → Dropout → LayerNorm
  Output :
    - user_emb    (N_users,    EMB_DIM)  — 128-dim user embeddings
    - article_emb (N_articles, EMB_DIM)  — 128-dim article embeddings

Click Prediction Head
---------------------
  score(u, n) = sigmoid(user_emb[u] · article_emb[n])

Usage
-----
  python src/models/gnn_encoder.py --demo     # quick forward pass test
  python src/models/gnn_encoder.py --train    # smoke test on synthetic data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

EMB_DIM = 128   # output embedding dimension


# ── Helper: feature dimension constants ──────────────────────────────────────

def _feat_dims() -> dict[str, int]:
    from src.graph.graph_utils import (USER_FEAT_DIM, ARTICLE_FEAT_DIM,
                                        TOPIC_FEAT_DIM, ENTITY_FEAT_DIM)
    return {
        "user":    USER_FEAT_DIM,
        "article": ARTICLE_FEAT_DIM,
        "topic":   TOPIC_FEAT_DIM,
        "entity":  ENTITY_FEAT_DIM,
    }


# ── GNN ───────────────────────────────────────────────────────────────────────

class HeteroGNNEncoder(nn.Module):
    """
    Two-layer heterogeneous GraphSAGE encoder.

    Parameters
    ----------
    feat_dims   : dict mapping node type → input feature dim
    emb_dim     : output embedding dimension (default 128)
    hidden_dim  : hidden dimension between layers (default 256)
    dropout     : dropout probability
    """

    def __init__(
        self,
        feat_dims:  Optional[dict[str, int]] = None,
        emb_dim:    int = EMB_DIM,
        hidden_dim: int = 256,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.emb_dim    = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout    = dropout

        fd = feat_dims or _feat_dims()

        try:
            from torch_geometric.nn import HeteroConv, SAGEConv  # type: ignore

            # ── Linear input projections (unify input dims to hidden_dim) ────
            self.input_proj = nn.ModuleDict({
                nt: nn.Linear(in_dim, hidden_dim)
                for nt, in_dim in fd.items()
            })

            # ── Layer 1: hidden_dim → hidden_dim ─────────────────────────────
            self.conv1 = HeteroConv({
                ("user",    "clicked",    "article"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("article", "belongs_to", "topic"):   SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("article", "mentions",   "entity"):  SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("topic",   "subtopic",   "topic"):   SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            }, aggr="sum")

            # ── Layer 2: hidden_dim → emb_dim ────────────────────────────────
            self.conv2 = HeteroConv({
                ("user",    "clicked",    "article"): SAGEConv((hidden_dim, hidden_dim), emb_dim),
                ("article", "belongs_to", "topic"):   SAGEConv((hidden_dim, hidden_dim), emb_dim),
                ("article", "mentions",   "entity"):  SAGEConv((hidden_dim, hidden_dim), emb_dim),
                ("topic",   "subtopic",   "topic"):   SAGEConv((hidden_dim, hidden_dim), emb_dim),
            }, aggr="sum")

            self._use_pyg = True
            log.info("GNN: using PyTorch Geometric HeteroConv + SAGEConv.")

        except ImportError:
            log.warning("PyTorch Geometric not available — using MLP fallback encoder.")
            self._use_pyg = False
            self.mlp_user    = nn.Sequential(nn.Linear(fd["user"],    hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))
            self.mlp_article = nn.Sequential(nn.Linear(fd["article"], hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))
            self.mlp_topic   = nn.Sequential(nn.Linear(fd["topic"],   hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))
            self.mlp_entity  = nn.Sequential(nn.Linear(fd["entity"],  hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))

        self.norm_user    = nn.LayerNorm(emb_dim)
        self.norm_article = nn.LayerNorm(emb_dim)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """
        Parameters
        ----------
        x_dict          : {node_type: feature_tensor}
        edge_index_dict : {(src, rel, dst): edge_index_tensor}

        Returns
        -------
        dict with 'user' and 'article' embedding tensors.
        """
        if self._use_pyg:
            # Project all node types to hidden_dim
            h = {nt: F.relu(self.input_proj[nt](x)) for nt, x in x_dict.items()}

            # Layer 1
            h = self.conv1(h, edge_index_dict)
            h = {nt: F.relu(F.dropout(x, p=self.dropout, training=self.training))
                 for nt, x in h.items()}

            # Layer 2
            h = self.conv2(h, edge_index_dict)
        else:
            h = {
                "user":    self.mlp_user(x_dict["user"]),
                "article": self.mlp_article(x_dict["article"]),
                "topic":   self.mlp_topic(x_dict.get("topic", torch.zeros(1, 40))),
                "entity":  self.mlp_entity(x_dict.get("entity", torch.zeros(1, 12))),
            }

        h["user"]    = self.norm_user(h["user"])
        h["article"] = self.norm_article(h["article"])
        return h


# ── Click prediction head ─────────────────────────────────────────────────────

class ClickPredictor(nn.Module):
    """
    Dot-product click predictor.

    score(u, n) = sigmoid(dot(user_emb[u], article_emb[n]))
    """

    def forward(self, user_emb: torch.Tensor, article_emb: torch.Tensor,
                user_idx: torch.Tensor, article_idx: torch.Tensor) -> torch.Tensor:
        u = user_emb[user_idx]        # (B, EMB_DIM)
        a = article_emb[article_idx]  # (B, EMB_DIM)
        return torch.sigmoid((u * a).sum(dim=-1))  # (B,)


class GNNRecommender(nn.Module):
    """Full model: encoder + click predictor."""

    def __init__(self, feat_dims: Optional[dict] = None,
                 emb_dim: int = EMB_DIM, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder   = HeteroGNNEncoder(feat_dims, emb_dim, hidden_dim, dropout)
        self.predictor = ClickPredictor()

    def forward(self, x_dict, edge_index_dict, user_idx, article_idx):
        emb   = self.encoder(x_dict, edge_index_dict)
        score = self.predictor(emb["user"], emb["article"], user_idx, article_idx)
        return score

    def encode(self, x_dict, edge_index_dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (user_emb, article_emb) for serving."""
        emb = self.encoder(x_dict, edge_index_dict)
        return emb["user"], emb["article"]


# ── Smoke test / demo ─────────────────────────────────────────────────────────

def _demo():
    """Forward-pass smoke test on random tensors."""
    from src.graph.graph_utils import USER_FEAT_DIM, ARTICLE_FEAT_DIM, TOPIC_FEAT_DIM, ENTITY_FEAT_DIM

    N_u, N_a, N_t, N_e = 50, 200, 20, 100
    x_dict = {
        "user":    torch.zeros(N_u, USER_FEAT_DIM),
        "article": torch.randn(N_a, ARTICLE_FEAT_DIM),
        "topic":   torch.randn(N_t, TOPIC_FEAT_DIM),
        "entity":  torch.randn(N_e, ENTITY_FEAT_DIM),
    }
    edge_index_dict = {
        ("user",    "clicked",    "article"): torch.randint(0, min(N_u, N_a), (2, 500)),
        ("article", "belongs_to", "topic"):   torch.tensor(
            [list(range(N_a)), [i % N_t for i in range(N_a)]], dtype=torch.long),
        ("article", "mentions",   "entity"):  torch.randint(0, min(N_a, N_e), (2, 300)),
        ("topic",   "subtopic",   "topic"):   torch.randint(0, N_t, (2, 15)),
    }
    # Fix edge indices to be in range
    for k, ei in edge_index_dict.items():
        src_type, _, dst_type = k
        n_src = x_dict[src_type].size(0)
        n_dst = x_dict[dst_type].size(0)
        ei[0] = ei[0] % n_src
        ei[1] = ei[1] % n_dst

    model = GNNRecommender()
    model.eval()

    with torch.no_grad():
        u_emb, a_emb = model.encode(x_dict, edge_index_dict)

    print(f"✅  Demo forward pass succeeded.")
    print(f"   user_emb   shape: {u_emb.shape}  ({N_u} users, {EMB_DIM}-dim)")
    print(f"   article_emb shape: {a_emb.shape}  ({N_a} articles, {EMB_DIM}-dim)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",  action="store_true")
    parser.add_argument("--train", action="store_true",
                        help="Smoke train on synthetic HeteroData for 2 epochs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    if args.demo:
        _demo()
    elif args.train:
        # Minimal smoke training
        from src.models.train_gnn import train
        train(epochs=2, use_synthetic=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
