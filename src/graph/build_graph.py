"""
build_graph.py
--------------
Constructs a heterogeneous knowledge graph from MIND data.

Node types:
  • User     — identified by user_id
  • Article  — identified by news_id
  • Topic    — (category, subcategory) pairs
  • Entity   — named entities extracted by spaCy NER

Edge types:
  • User    → CLICKED    → Article
  • Article → BELONGS_TO → Topic
  • Article → MENTIONS   → Entity
  • Topic   → SUBTOPIC   → Topic  (category → subcategory)
  • Entity  → CO_OCCURS  → Entity (within same article)

Output
------
  checkpoints/news_graph.pkl  — pickled NetworkX MultiDiGraph
  checkpoints/graph_meta.json — node/edge counts + ID mappings

Usage
-----
  python src/graph/build_graph.py [--split train] [--co-occur]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import networkx as nx

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
PROCESSED   = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(split: str = "train", add_co_occur: bool = False) -> nx.MultiDiGraph:
    """
    Build and return the heterogeneous graph.

    Parameters
    ----------
    split       : MIND split to use ('train' or 'dev')
    add_co_occur: Whether to add Entity–CO_OCCURS–Entity edges
                  (expensive for large graphs)
    """
    G = nx.MultiDiGraph()

    articles_path = PROCESSED / split / "articles_scored.jsonl"
    users_path    = PROCESSED / split / "users.json"

    if not articles_path.exists():
        articles_path = PROCESSED / split / "articles.jsonl"  # fallback
    if not articles_path.exists():
        log.error("No article file found. Run preprocess.py first.")
        return G

    # ── 1. Load and add Article nodes ────────────────────────────────────────
    log.info("Loading articles from %s …", articles_path)
    articles: dict[str, dict] = {}
    with open(articles_path, encoding="utf-8") as fh:
        for line in fh:
            a = json.loads(line)
            nid = a["news_id"]
            articles[nid] = a
            G.add_node(
                f"article:{nid}",
                node_type="article",
                news_id=nid,
                category=a.get("category", ""),
                subcategory=a.get("subcategory", ""),
                title=a.get("title", ""),
                persuasion_risk=a.get("persuasion_risk", 0.0),
            )

    log.info("  Added %d article nodes.", len(articles))

    # ── 2. Add Topic nodes and Article→Topic edges ────────────────────────────
    log.info("Adding topic nodes …")
    topic_count = 0
    for nid, a in articles.items():
        cat    = a.get("category", "unknown").lower()
        subcat = a.get("subcategory", "unknown").lower()

        cat_node    = f"topic:{cat}"
        subcat_node = f"topic:{cat}/{subcat}"

        if not G.has_node(cat_node):
            G.add_node(cat_node, node_type="topic", level="category", name=cat)
            topic_count += 1
        if not G.has_node(subcat_node):
            G.add_node(subcat_node, node_type="topic", level="subcategory", name=subcat)
            topic_count += 1
            # Topic hierarchy edge
            G.add_edge(cat_node, subcat_node, edge_type="SUBTOPIC")

        G.add_edge(f"article:{nid}", subcat_node, edge_type="BELONGS_TO")

    log.info("  Added %d topic nodes.", topic_count)

    # ── 3. Add Entity nodes and Article→Entity edges ──────────────────────────
    log.info("Adding entity nodes …")
    entity_count = 0
    article_entities: dict[str, list[str]] = defaultdict(list)

    for nid, a in articles.items():
        entities = a.get("entities", [])
        seen_ents = set()
        for ent in entities:
            ent_key = f"entity:{ent['label']}:{ent['text'].lower()}"
            if not G.has_node(ent_key):
                G.add_node(ent_key, node_type="entity",
                           entity_text=ent["text"], entity_label=ent["label"])
                entity_count += 1
            if ent_key not in seen_ents:
                G.add_edge(f"article:{nid}", ent_key, edge_type="MENTIONS")
                seen_ents.add(ent_key)
            article_entities[nid].append(ent_key)

    log.info("  Added %d entity nodes.", entity_count)

    # ── 4. Co-occurrence edges (Entity↔Entity within same article) ────────────
    if add_co_occur:
        log.info("Adding entity co-occurrence edges …")
        cooc_count = 0
        for nid, ent_list in article_entities.items():
            for e1, e2 in combinations(set(ent_list), 2):
                G.add_edge(e1, e2, edge_type="CO_OCCURS", article=nid)
                cooc_count += 1
        log.info("  Added %d co-occurrence edges.", cooc_count)

    # ── 5. Load users and add User→Article (CLICKED) edges ───────────────────
    if users_path.exists():
        log.info("Adding user nodes from %s …", users_path)
        with open(users_path, encoding="utf-8") as fh:
            users = json.load(fh)

        user_count = 0
        click_count = 0
        for uid, data in users.items():
            user_node = f"user:{uid}"
            G.add_node(user_node, node_type="user", user_id=uid)
            user_count += 1
            # `history` contains previously read items; `impressions` contains
            # clicked items extracted during preprocessing.
            for nid in data.get("history", []):
                art_node = f"article:{nid}"
                if G.has_node(art_node):
                    G.add_edge(user_node, art_node, edge_type="VIEWED")
                    click_count += 1
            for nid in data.get("impressions", []):
                art_node = f"article:{nid}"
                if G.has_node(art_node):
                    G.add_edge(user_node, art_node, edge_type="CLICKED")
                    click_count += 1
        log.info("  Added %d user nodes + %d CLICKED edges.", user_count, click_count)
    else:
        log.warning("users.json not found — no user nodes added.")

    return G


def save_graph(G: nx.MultiDiGraph, suffix: str = ""):
    """Pickle the graph and write metadata JSON."""
    pkl_path  = CHECKPOINTS / f"news_graph{suffix}.pkl"
    meta_path = CHECKPOINTS / f"graph_meta{suffix}.json"

    with open(pkl_path, "wb") as fh:
        pickle.dump(G, fh, protocol=4)
    log.info("✅  Graph saved → %s  (%d nodes, %d edges)",
             pkl_path, G.number_of_nodes(), G.number_of_edges())

    # Node type breakdown
    type_counts: dict[str, int] = defaultdict(int)
    for _, data in G.nodes(data=True):
        type_counts[data.get("node_type", "unknown")] += 1

    edge_type_counts: dict[str, int] = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_type_counts[data.get("edge_type", "unknown")] += 1

    meta = {
        "num_nodes":        G.number_of_nodes(),
        "num_edges":        G.number_of_edges(),
        "node_type_counts": dict(type_counts),
        "edge_type_counts": dict(edge_type_counts),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("   Metadata → %s", meta_path)
    return pkl_path


def load_graph(suffix: str = "") -> nx.MultiDiGraph:
    pkl_path = CHECKPOINTS / f"news_graph{suffix}.pkl"
    if not pkl_path.exists():
        log.error("Graph not found at %s. Run build_graph.py first.", pkl_path)
        return nx.MultiDiGraph()
    with open(pkl_path, "rb") as fh:
        G = pickle.load(fh)
    log.info("Loaded graph: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())
    return G


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build heterogeneous MIND knowledge graph.")
    parser.add_argument("--split",     default="train", choices=["train", "dev"])
    parser.add_argument("--co-occur",  action="store_true",
                        help="Add entity co-occurrence edges (slower).")
    args = parser.parse_args()

    log.info("Building graph for split=%s …", args.split)
    G = build_graph(split=args.split, add_co_occur=args.co_occur)
    save_graph(G)
    log.info("Graph construction complete.")


if __name__ == "__main__":
    main()
