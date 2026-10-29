"""
init_db.py  —  Database Initializer & Seeder
-------------------------------------------
Populates SQLite with initial article corpus and default user profiles
so the system is immediately fully operational.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.data.db import ArticleModel, InteractionModel, SessionLocal, UserModel, init_db
from src.ranking.reranker import Reranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def seed_database(force: bool = False):
    """Seed database with default articles and users if empty."""
    init_db()
    session = SessionLocal()

    try:
        article_count = session.query(ArticleModel).count()
        if article_count == 0 or force:
            log.info("Seeding articles into database...")
            articles_to_insert = []

            # Check if processed articles exist
            loaded_from_file = False
            for split in ("train", "dev"):
                path = PROCESSED / split / "articles_scored.jsonl"
                if path.exists():
                    with open(path, encoding="utf-8") as fh:
                        for line in fh:
                            a = json.loads(line)
                            articles_to_insert.append(
                                ArticleModel(
                                    news_id=a["news_id"],
                                    title=a.get("title", ""),
                                    abstract=a.get("abstract", ""),
                                    category=a.get("category", "general"),
                                    subcategory=a.get("subcategory", "general"),
                                    url=a.get("url", ""),
                                    persuasion_risk=float(a.get("persuasion_risk", 0.0)),
                                    persuasion_scores_json=json.dumps(a.get("persuasion_scores", {})),
                                )
                            )
                    log.info("Loaded %d articles from %s", len(articles_to_insert), path)
                    loaded_from_file = True
                    break

            if not loaded_from_file:
                # Seed with demo articles from Reranker
                demo_arts = Reranker._demo_articles()
                for nid, a in demo_arts.items():
                    articles_to_insert.append(
                        ArticleModel(
                            news_id=nid,
                            title=a["title"],
                            abstract=a["abstract"],
                            category=a["category"],
                            subcategory=a["subcategory"],
                            url=a["url"],
                            persuasion_risk=a["persuasion_risk"],
                            persuasion_scores_json=json.dumps(a["persuasion_scores"]),
                        )
                    )
                log.info("Seeded %d demo articles", len(articles_to_insert))

            session.bulk_save_objects(articles_to_insert)
            session.commit()
            log.info("✅ Database now contains %d articles.", session.query(ArticleModel).count())

        # Seed standard users
        user_count = session.query(UserModel).count()
        if user_count == 0 or force:
            log.info("Seeding standard demo users...")
            default_users = [
                UserModel(
                    user_id="user_demo",
                    susceptibility=0.42,
                    n_updates=87,
                    vector_json=json.dumps([
                        0.12, 0.65, 0.31, 0.18, 0.22, 0.44, 0.15, 0.08, 0.10,
                        0.72, 0.55, 0.06, 0.28, 0.19, 0.14, 0.11, 0.33, 0.09,
                        0.21, 0.38, 0.17, 0.13, 0.07,
                    ]),
                ),
                UserModel(
                    user_id="user_sensitive",
                    susceptibility=0.76,
                    n_updates=142,
                    vector_json=json.dumps([
                        0.30, 0.88, 0.65, 0.50, 0.40, 0.82, 0.30, 0.20, 0.20,
                        0.94, 0.81, 0.10, 0.40, 0.20, 0.30, 0.20, 0.50, 0.20,
                        0.40, 0.60, 0.30, 0.20, 0.10,
                    ]),
                ),
                UserModel(
                    user_id="user_skeptic",
                    susceptibility=0.15,
                    n_updates=210,
                    vector_json=json.dumps([
                        0.25, 0.08, 0.05, 0.10, 0.12, 0.15, 0.05, 0.04, 0.06,
                        0.10, 0.08, 0.05, 0.78, 0.05, 0.04, 0.05, 0.10, 0.05,
                        0.08, 0.06, 0.05, 0.05, 0.05,
                    ]),
                ),
            ]
            session.bulk_save_objects(default_users)
            session.commit()
            log.info("✅ Database seeded with %d standard user personas.", len(default_users))

    except Exception as exc:
        session.rollback()
        log.exception("Error seeding database: %s", exc)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed_database()
