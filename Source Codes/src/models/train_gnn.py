"""
train_gnn.py
------------
Colab-ready training script for the heterogeneous GraphSAGE recommender.

Algorithm
---------
  Mini-batch Bayesian Personalized Ranking (BPR) loss:
    L = -E[ log σ(score(u, pos) - score(u, neg)) ]

  For each positive (user, clicked_article) pair we sample one
  negative (user, random_unclicked_article) pair.

Metrics tracked per epoch
--------------------------
  • BPR loss (train)
  • AUC on validation impression pairs
  • nDCG@10 on validation impression groups

Usage
-----
  # Local (smoke test — no MIND needed):
  python src/models/train_gnn.py --epochs 2 --synthetic

  # Local (MINDsmall):
  python src/models/train_gnn.py --epochs 10

  # Google Colab (MINDlarge, A100 40 GB):
  python src/models/train_gnn.py --epochs 20 --batch-size 4096 --hidden 512

Checkpoints
-----------
  checkpoints/gnn_model.pt        — best model weights
  checkpoints/gnn_train_log.jsonl — per-epoch metrics log
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
PROCESSED   = ROOT / "data" / "processed"
CHECKPOINTS.mkdir(exist_ok=True)


# ── BPR Dataset ───────────────────────────────────────────────────────────────

class BPRDataset(Dataset):
    """
    Each item is (user_int_id, pos_article_int_id, neg_article_int_id).
    Built from the CLICKED edges in the HeteroData object.
    """

    def __init__(self, user_idx: dict, article_idx: dict, edge_index: torch.Tensor,
                 neg_ratio: int = 1):
        # edge_index shape: (2, num_clicks) — [user_ids, article_ids]
        self.pos_pairs = list(zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
        ))
        self.n_articles = len(article_idx)
        self.neg_ratio  = neg_ratio

        # Build set of clicked articles per user for hard-negative avoidance
        self.user_clicks: dict[int, set] = {}
        for u, a in self.pos_pairs:
            self.user_clicks.setdefault(u, set()).add(a)

    def __len__(self):
        return len(self.pos_pairs) * self.neg_ratio

    def __getitem__(self, idx: int):
        u, pos = self.pos_pairs[idx % len(self.pos_pairs)]
        # Sample hard negative (not in user's click history)
        neg = random.randint(0, self.n_articles - 1)
        clicked = self.user_clicks.get(u, set())
        attempts = 0
        while neg in clicked and attempts < 10:
            neg = random.randint(0, self.n_articles - 1)
            attempts += 1
        return torch.tensor(u), torch.tensor(pos), torch.tensor(neg)


# ── AUC / nDCG evaluation ─────────────────────────────────────────────────────

def _evaluate(user_emb: torch.Tensor, article_emb: torch.Tensor,
              behaviors_path: Path, article_idx: dict, user_idx: dict,
              k: int = 10) -> dict:
    """
    Evaluate on held-out MIND behaviors (impression list + ground-truth click).
    Returns: {"auc": float, "ndcg@k": float}
    """
    if not behaviors_path.exists():
        return {}

    import csv
    from sklearn.metrics import roc_auc_score

    aucs, ndcgs = [], []
    with open(behaviors_path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            uid         = row[1]
            impressions = row[4].split()
            if len(impressions) < 2:
                continue

            u_int = user_idx.get(f"user:{uid}")
            if u_int is None:
                continue

            u_vec    = user_emb[u_int]
            labels, scores = [], []
            for imp in impressions:
                parts = imp.split("-")
                if len(parts) != 2:
                    continue
                nid, clicked = parts[0], int(parts[1])
                a_int = article_idx.get(f"article:{nid}")
                if a_int is None:
                    continue
                score = float((u_vec * article_emb[a_int]).sum().item())
                labels.append(clicked)
                scores.append(score)

            if len(set(labels)) < 2:
                continue   # skip if no negative or no positive

            # AUC
            auc = roc_auc_score(labels, scores)
            aucs.append(auc)

            # nDCG@k
            paired    = sorted(zip(scores, labels), reverse=True)
            dcg       = sum(lab / np.log2(i + 2) for i, (_, lab) in enumerate(paired[:k]))
            ideal_lab = sorted(labels, reverse=True)
            idcg      = sum(lab / np.log2(i + 2) for i, lab in enumerate(ideal_lab[:k]))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "auc":    float(np.mean(aucs))   if aucs   else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


# ── Synthetic HeteroData ──────────────────────────────────────────────────────

def _synthetic_hetero_data():
    """Build tiny HeteroData for smoke-testing without MIND."""
    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        return None, {}, {}

    from src.graph.graph_utils import (USER_FEAT_DIM, ARTICLE_FEAT_DIM,
                                        TOPIC_FEAT_DIM, ENTITY_FEAT_DIM)

    N_u, N_a, N_t, N_e = 100, 500, 20, 80
    data = HeteroData()
    data["user"].x    = torch.zeros(N_u, USER_FEAT_DIM)
    data["article"].x = torch.randn(N_a, ARTICLE_FEAT_DIM)
    data["topic"].x   = torch.randn(N_t, TOPIC_FEAT_DIM)
    data["entity"].x  = torch.randn(N_e, ENTITY_FEAT_DIM)
    data["user"].num_nodes    = N_u
    data["article"].num_nodes = N_a
    data["topic"].num_nodes   = N_t
    data["entity"].num_nodes  = N_e

    n_clicks = 2000
    click_u  = torch.randint(0, N_u, (n_clicks,))
    click_a  = torch.randint(0, N_a, (n_clicks,))
    data["user", "clicked", "article"].edge_index = torch.stack([click_u, click_a])

    art_topics = torch.stack([torch.arange(N_a), torch.arange(N_a) % N_t])
    data["article", "belongs_to", "topic"].edge_index = art_topics

    art_ents = torch.stack([torch.randint(0, N_a, (500,)), torch.randint(0, N_e, (500,))])
    data["article", "mentions", "entity"].edge_index = art_ents

    data["topic", "subtopic", "topic"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    user_idx    = {f"user:U{i:05d}": i for i in range(N_u)}
    article_idx = {f"article:N{i:06d}": i for i in range(N_a)}
    return data, user_idx, article_idx


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    epochs:        int   = 10,
    batch_size:    int   = 512,
    lr:            float = 1e-3,
    hidden:        int   = 256,
    emb_dim:       int   = 128,
    dropout:       float = 0.2,
    eval_k:        int   = 10,
    use_synthetic: bool  = False,
    wandb_log:     bool  = False,
):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on: %s", device)

    # ── Load / build data ─────────────────────────────────────────────────────
    if use_synthetic:
        log.info("Using synthetic HeteroData …")
        hetero_data, user_idx, article_idx = _synthetic_hetero_data()
        if hetero_data is None:
            log.error("PyG not available for synthetic data.")
            return
        behaviors_val = None
    else:
        from src.graph.graph_utils import load_hetero_data
        hetero_data, user_idx, article_idx = load_hetero_data()
        behaviors_val = ROOT / "data" / "raw" / "mind" / "dev" / "behaviors.tsv"

    # ── Build BPR dataset ─────────────────────────────────────────────────────
    click_ei = hetero_data["user", "clicked", "article"].edge_index
    bpr_ds   = BPRDataset(user_idx, article_idx, click_ei)
    bpr_dl   = DataLoader(bpr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    log.info("BPR dataset: %d positive pairs, %d batches", len(bpr_ds), len(bpr_dl))

    # ── Build model ───────────────────────────────────────────────────────────
    from src.models.gnn_encoder import GNNRecommender
    feat_dims = {nt: hetero_data[nt].x.size(1) for nt in ["user", "article", "topic", "entity"]}
    model     = GNNRecommender(feat_dims=feat_dims, emb_dim=emb_dim,
                                hidden_dim=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Move graph data to device
    x_dict          = {k: v.x.to(device) for k, v in hetero_data.node_items()}
    edge_index_dict = {k: v.edge_index.to(device) for k, v in hetero_data.edge_items()}

    # ── W&B ──────────────────────────────────────────────────────────────────
    if wandb_log:
        try:
            import wandb
            wandb.init(project="responsible-news-recommender", config=dict(
                epochs=epochs, batch_size=batch_size, lr=lr, hidden=hidden,
                emb_dim=emb_dim, dropout=dropout,
            ))
        except Exception:
            log.warning("W&B not available — disabling tracking.")
            wandb_log = False

    # ── Training ──────────────────────────────────────────────────────────────
    best_auc   = 0.0
    ckpt_path  = CHECKPOINTS / "gnn_model.pt"
    log_path   = CHECKPOINTS / "gnn_train_log.jsonl"

    with open(log_path, "w") as logf:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches  = 0

            for u_idx, pos_idx, neg_idx in tqdm(bpr_dl, desc=f"Epoch {epoch}/{epochs}"):
                u_idx   = u_idx.to(device)
                pos_idx = pos_idx.to(device)
                neg_idx = neg_idx.to(device)

                optimizer.zero_grad()

                # Full-graph encoding
                u_emb, a_emb = model.encode(x_dict, edge_index_dict)

                pos_score = (u_emb[u_idx] * a_emb[pos_idx]).sum(dim=-1)
                neg_score = (u_emb[u_idx] * a_emb[neg_idx]).sum(dim=-1)

                # BPR loss
                loss = -F.logsigmoid(pos_score - neg_score).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # ── Evaluate ──────────────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                u_emb, a_emb = model.encode(x_dict, edge_index_dict)

            val_metrics = {}
            if behaviors_val and behaviors_val.exists():
                val_metrics = _evaluate(
                    u_emb.cpu(), a_emb.cpu(), behaviors_val,
                    article_idx, user_idx, k=eval_k,
                )

            metrics = {"epoch": epoch, "loss": avg_loss, **val_metrics}
            logf.write(json.dumps(metrics) + "\n")
            log.info("Epoch %d | loss=%.4f | %s", epoch, avg_loss,
                     " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

            if wandb_log:
                import wandb
                wandb.log(metrics)

            # ── Save best ─────────────────────────────────────────────────────
            auc = val_metrics.get("auc", -1.0)
            if auc > best_auc or epoch == 1:
                best_auc = auc
                torch.save({
                    "model_state":   model.state_dict(),
                    "user_idx":      user_idx,
                    "article_idx":   article_idx,
                    "emb_dim":       emb_dim,
                    "hidden_dim":    hidden,
                }, ckpt_path)
                log.info("  ✅  Best model saved (AUC=%.4f) → %s", best_auc, ckpt_path)

    if wandb_log:
        import wandb
        wandb.finish()

    log.info("Training complete. Checkpoint: %s", ckpt_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE news recommender.")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--hidden",      type=int,   default=256)
    parser.add_argument("--emb-dim",     type=int,   default=128)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic data (no MIND required)")
    parser.add_argument("--wandb",       action="store_true")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        use_synthetic=args.synthetic,
        wandb_log=args.wandb,
    )


if __name__ == "__main__":
    main()
