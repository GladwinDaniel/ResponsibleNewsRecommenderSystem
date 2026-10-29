"""
generate_synthetic_data.py
--------------------------
Generates a MIND-format synthetic dataset so the full pipeline can be
developed and tested without downloading the real MIND dataset.

Output layout (mirrors real MIND):
  data/raw/mind/train/news.tsv
  data/raw/mind/train/behaviors.tsv
  data/raw/mind/dev/news.tsv
  data/raw/mind/dev/behaviors.tsv
"""

import argparse
import csv
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIND_TRAIN = ROOT / "data" / "raw" / "mind" / "train"
MIND_DEV   = ROOT / "data" / "raw" / "mind" / "dev"

# ── Fake content pools ────────────────────────────────────────────────────────

CATEGORIES = ["politics", "world", "health", "technology", "science",
               "sports", "entertainment", "finance", "lifestyle", "travel"]

SUBCATS = {
    "politics":      ["elections", "government", "policy", "congress"],
    "world":         ["europe", "asia", "africa", "middleeast"],
    "health":        ["medicine", "wellness", "mentalhealth", "nutrition"],
    "technology":    ["ai", "cybersecurity", "gadgets", "software"],
    "science":       ["climate", "space", "biology", "physics"],
    "sports":        ["football", "cricket", "tennis", "olympic"],
    "entertainment": ["movies", "music", "celebrity", "streaming"],
    "finance":       ["markets", "crypto", "economy", "investment"],
    "lifestyle":     ["food", "travel", "fashion", "parenting"],
    "travel":        ["destinations", "budget", "luxury", "adventure"],
}

PROVIDERS = ["reuters", "bbc", "cnn", "nytimes", "guardian",
             "foxnews", "apnews", "bloomberg", "wsj", "theverge"]

# Title templates — mixing neutral and clickbait-style
TITLE_TEMPLATES = [
    "{Category} report: new findings challenge {topic} assumptions",
    "SHOCKING: {Actor} reveals hidden truth about {topic}",
    "Scientists discover breakthrough in {topic} research",
    "You won't believe what {Actor} said about {topic}",
    "{Country} announces major policy shift on {topic}",
    "Experts warn: {topic} crisis is worse than reported",
    "New study: {topic} linked to {outcome}",
    "{Topic} update: everything you need to know",
    "Opinion: why {topic} matters more than ever",
    "Investigation: how {topic} is being manipulated",
]

ACTORS   = ["government", "CEO", "scientists", "lawmakers", "experts",
            "officials", "researchers", "celebrities", "leaders", "whistleblower"]
TOPICS   = ["climate change", "AI regulation", "public health", "immigration",
            "election integrity", "stock markets", "social media", "education",
            "housing crisis", "energy policy"]
OUTCOMES = ["job losses", "health risks", "economic growth", "policy changes",
            "social unrest", "innovation", "inequality", "political gridlock"]
COUNTRIES = ["USA", "UK", "India", "China", "EU", "Brazil", "Australia",
             "Germany", "France", "Japan"]


def _rand_title() -> str:
    tpl = random.choice(TITLE_TEMPLATES)
    return tpl.format(
        Category=random.choice(CATEGORIES).title(),
        topic=random.choice(TOPICS),
        Actor=random.choice(ACTORS).title(),
        Topic=random.choice(TOPICS).title(),
        Country=random.choice(COUNTRIES),
        outcome=random.choice(OUTCOMES),
    )


def _rand_abstract() -> str:
    sentences = [
        f"A new report on {random.choice(TOPICS)} has raised concerns among policymakers.",
        f"Researchers at a leading university found significant correlations in {random.choice(TOPICS)}.",
        f"Officials in {random.choice(COUNTRIES)} announced steps to address {random.choice(TOPICS)}.",
        f"The latest data shows a {random.randint(5, 45)}% change in metrics related to {random.choice(TOPICS)}.",
        f"Critics argue that the response to {random.choice(TOPICS)} has been inadequate.",
    ]
    return " ".join(random.sample(sentences, k=random.randint(2, 4)))


# ── News TSV writer ───────────────────────────────────────────────────────────

def generate_news(n: int = 1000, seed: int = 42) -> list[dict]:
    random.seed(seed)
    articles = []
    for i in range(n):
        cat = random.choice(CATEGORIES)
        subcat = random.choice(SUBCATS[cat])
        provider = random.choice(PROVIDERS)
        news_id = f"N{i:06d}"
        articles.append({
            "news_id":    news_id,
            "category":   cat,
            "subcategory": subcat,
            "title":      _rand_title(),
            "abstract":   _rand_abstract(),
            "url":        f"https://{provider}.com/news/{uuid.uuid4().hex[:8]}",
            "title_entities": json.dumps([]),
            "abstract_entities": json.dumps([]),
        })
    return articles


def write_news_tsv(articles: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["news_id", "category", "subcategory", "title", "abstract",
            "url", "title_entities", "abstract_entities"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t",
                                extrasaction="ignore")
        for row in articles:
            writer.writerow(row)
    print(f"  Wrote {len(articles)} articles → {path}")


# ── Behaviors TSV writer ──────────────────────────────────────────────────────

def generate_behaviors(articles: list[dict], n_users: int = 500,
                       n_impressions: int = 5000, seed: int = 42) -> list[dict]:
    random.seed(seed + 1)
    news_ids = [a["news_id"] for a in articles]
    base_time = datetime(2023, 6, 1)
    rows = []

    for imp_id in range(n_impressions):
        user_id = f"U{random.randint(0, n_users - 1):05d}"
        time_str = (base_time + timedelta(minutes=random.randint(0, 60 * 24 * 30))).strftime(
            "%m/%d/%Y %I:%M:%S %p"
        )
        history = " ".join(random.sample(news_ids, k=min(10, len(news_ids))))
        # impression: 5–15 candidate articles, one clicked
        cands = random.sample(news_ids, k=random.randint(5, 15))
        clicked = random.choice(cands)
        impression_strs = [f"{nid}-{1 if nid == clicked else 0}" for nid in cands]
        rows.append({
            "impression_id": str(imp_id),
            "user_id":       user_id,
            "time":          time_str,
            "history":       history,
            "impressions":   " ".join(impression_strs),
        })
    return rows


def write_behaviors_tsv(behaviors: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t",
                                extrasaction="ignore")
        for row in behaviors:
            writer.writerow(row)
    print(f"  Wrote {len(behaviors)} impressions → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MIND-format data.")
    parser.add_argument("--n-articles",    type=int, default=1000)
    parser.add_argument("--n-users",       type=int, default=500)
    parser.add_argument("--n-impressions", type=int, default=5000)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    print("Generating synthetic MIND-format dataset…")

    # Training split: full articles, 80 % of impressions
    train_articles  = generate_news(args.n_articles, seed=args.seed)
    train_behaviors = generate_behaviors(train_articles, args.n_users,
                                         int(args.n_impressions * 0.8), args.seed)

    write_news_tsv(train_articles, MIND_TRAIN / "news.tsv")
    write_behaviors_tsv(train_behaviors, MIND_TRAIN / "behaviors.tsv")

    # Dev split: same articles, 20 % of impressions
    dev_articles  = generate_news(args.n_articles, seed=args.seed + 100)
    dev_behaviors = generate_behaviors(dev_articles, args.n_users,
                                       int(args.n_impressions * 0.2), args.seed + 100)

    write_news_tsv(dev_articles, MIND_DEV / "news.tsv")
    write_behaviors_tsv(dev_behaviors, MIND_DEV / "behaviors.tsv")

    print("✅  Synthetic data ready. You can now run preprocess.py.")


if __name__ == "__main__":
    main()
