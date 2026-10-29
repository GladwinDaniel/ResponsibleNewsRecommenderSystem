"""
graph_utils.py
--------------
Converts a NetworkX MultiDiGraph (from build_graph.py) into a
PyTorch Geometric HeteroData object for GNN training.

Node feature encoding
---------------------
  user    → zero vector (embeddings learned during training)
  article → [persuasion_risk(1), category_onehot(10), subcat_onehot(30)] → 41-dim
  topic   → category/subcat one-hot
  entity  → NER label one-hot (PERSON, ORG, GPE, …)

Usage
-----
  from src.graph.graph_utils import hetero_from_nx
  data = hetero_from_nx(G)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"

# ── Feature constants ─────────────────────────────────────────────────────────

CATEGORIES = ["politics", "world", "health", "technology", "science",
               "sports", "entertainment", "finance", "lifestyle", "travel"]

SUBCATEGORIES = [
    "elections", "government", "policy", "congress",
    "europe", "asia", "africa", "middleeast",
    "medicine", "wellness", "mentalhealth", "nutrition",
    "ai", "cybersecurity", "gadgets", "software",
    "climate", "space", "biology", "physics",
    "football", "cricket", "tennis", "olympic",
    "movies", "music", "celebrity", "streaming",
    "markets", "crypto",
]

ENTITY_LABELS = ["PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT",
                 "WORK_OF_ART", "DATE", "TIME", "MONEY", "PERCENT", "NORP"]

USER_FEAT_DIM    = 16    # learned embedding dimension init
ARTICLE_FEAT_DIM = 1 + len(CATEGORIES) + len(SUBCATEGORIES)   # 41
TOPIC_FEAT_DIM   = len(CATEGORIES) + len(SUBCATEGORIES)        # 40
ENTITY_FEAT_DIM  = len(ENTITY_LABELS)                          # 12


def _cat_onehot(value: str, vocab: list[str]) -> list[float]:
    vec = [0.0] * len(vocab)
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    return vec


# ── Main converter ────────────────────────────────────────────────────────────

def hetero_from_nx(G) -> "HeteroData":  # type: ignore
    """
    Convert a NetworkX MultiDiGraph to a PyTorch Geometric HeteroData object.

    Returns
    -------
    torch_geometric.data.HeteroData with:
      - node stores:  'user', 'article', 'topic', 'entity'
      - edge stores:  ('user','clicked','article'),
                      ('article','belongs_to','topic'),
                      ('article','mentions','entity'),
                      ('topic','subtopic','topic')
    Also returns three dicts mapping node-type → node_id → integer index.
    """
    try:
        from torch_geometric.data import HeteroData  # type: ignore
    except ImportError:
        log.error("torch_geometric not installed. Run: pip install torch-geometric")
        raise

    data = HeteroData()

    # ── Build node index maps ─────────────────────────────────────────────────
    user_nodes    = [n for n, d in G.nodes(data=True) if d.get("node_type") == "user"]
    article_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "article"]
    topic_nodes   = [n for n, d in G.nodes(data=True) if d.get("node_type") == "topic"]
    entity_nodes  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]

    user_idx    = {n: i for i, n in enumerate(user_nodes)}
    article_idx = {n: i for i, n in enumerate(article_nodes)}
    topic_idx   = {n: i for i, n in enumerate(topic_nodes)}
    entity_idx  = {n: i for i, n in enumerate(entity_nodes)}

    # ── Node features ─────────────────────────────────────────────────────────

    # User: learnable → init as zeros, embedding learned in GNN
    data["user"].x = torch.zeros(len(user_nodes), USER_FEAT_DIM, dtype=torch.float32)
    data["user"].num_nodes = len(user_nodes)

    # Article: [risk, category_onehot, subcat_onehot]
    art_feats = []
    for n in article_nodes:
        d    = G.nodes[n]
        risk = float(d.get("persuasion_risk", 0.0))
        cat  = _cat_onehot(d.get("category", ""), CATEGORIES)
        sub  = _cat_onehot(d.get("subcategory", ""), SUBCATEGORIES)
        art_feats.append([risk] + cat + sub)
    data["article"].x = torch.tensor(art_feats, dtype=torch.float32)
    data["article"].num_nodes = len(article_nodes)

    # Topic: [category_onehot, subcat_onehot]
    topic_feats = []
    for n in topic_nodes:
        d    = G.nodes[n]
        name = d.get("name", "")
        cat  = _cat_onehot(name, CATEGORIES)
        sub  = _cat_onehot(name, SUBCATEGORIES)
        topic_feats.append(cat + sub)
    data["topic"].x = torch.tensor(topic_feats, dtype=torch.float32) if topic_feats else \
                      torch.zeros(len(topic_nodes), TOPIC_FEAT_DIM)
    data["topic"].num_nodes = len(topic_nodes)

    # Entity: NER label one-hot
    entity_feats = []
    for n in entity_nodes:
        d   = G.nodes[n]
        lbl = _cat_onehot(d.get("entity_label", ""), ENTITY_LABELS)
        entity_feats.append(lbl)
    data["entity"].x = torch.tensor(entity_feats, dtype=torch.float32) if entity_feats else \
                       torch.zeros(len(entity_nodes), ENTITY_FEAT_DIM)
    data["entity"].num_nodes = len(entity_nodes)

    # ── Edge index tensors ────────────────────────────────────────────────────

    clicked_src, clicked_dst           = [], []
    belongs_src, belongs_dst           = [], []
    mentions_src, mentions_dst         = [], []
    subtopic_src, subtopic_dst         = [], []

    for u, v, d in G.edges(data=True):
        et = d.get("edge_type", "")

        if et == "CLICKED" and u in user_idx and v in article_idx:
            clicked_src.append(user_idx[u])
            clicked_dst.append(article_idx[v])

        elif et == "BELONGS_TO" and u in article_idx and v in topic_idx:
            belongs_src.append(article_idx[u])
            belongs_dst.append(topic_idx[v])

        elif et == "MENTIONS" and u in article_idx and v in entity_idx:
            mentions_src.append(article_idx[u])
            mentions_dst.append(entity_idx[v])

        elif et == "SUBTOPIC" and u in topic_idx and v in topic_idx:
            subtopic_src.append(topic_idx[u])
            subtopic_dst.append(topic_idx[v])

    def _edge_index(src_list, dst_list) -> torch.Tensor:
        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor([src_list, dst_list], dtype=torch.long)

    data["user",    "clicked",    "article"].edge_index = _edge_index(clicked_src, clicked_dst)
    data["article", "belongs_to", "topic"  ].edge_index = _edge_index(belongs_src, belongs_dst)
    data["article", "mentions",   "entity" ].edge_index = _edge_index(mentions_src, mentions_dst)
    data["topic",   "subtopic",   "topic"  ].edge_index = _edge_index(subtopic_src, subtopic_dst)

    log.info(
        "HeteroData: users=%d, articles=%d, topics=%d, entities=%d | "
        "clicked=%d, belongs_to=%d, mentions=%d, subtopic=%d",
        len(user_nodes), len(article_nodes), len(topic_nodes), len(entity_nodes),
        len(clicked_src), len(belongs_src), len(mentions_src), len(subtopic_src),
    )

    return data, user_idx, article_idx


# ── Save / load ───────────────────────────────────────────────────────────────

def save_hetero_data(data, user_idx: dict, article_idx: dict, suffix: str = ""):
    path = CHECKPOINTS / f"hetero_data{suffix}.pt"
    torch.save({
        "data":        data,
        "user_idx":    user_idx,
        "article_idx": article_idx,
    }, path)
    log.info("✅  HeteroData saved → %s", path)
    return path


def load_hetero_data(suffix: str = ""):
    path = CHECKPOINTS / f"hetero_data{suffix}.pt"
    payload = torch.load(path, map_location="cpu")
    return payload["data"], payload["user_idx"], payload["article_idx"]


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.graph.build_graph import load_graph
    log.basicConfig = logging.basicConfig
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    G = load_graph()
    if G.number_of_nodes() == 0:
        log.error("Empty graph — run build_graph.py first.")
    else:
        data, user_idx, article_idx = hetero_from_nx(G)
        save_hetero_data(data, user_idx, article_idx)
        print("Done.")
