"""
db.py  —  Database Engine & SQLAlchemy Models for Responsible News Recommender
------------------------------------------------------------------------------
Provides persistent storage for:
  - Users & their 23-dimensional persuasion profiles
  - News articles with persuasion scores & metadata
  - User interactions (clicks, skips, and sensationalism flags)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "responsible_news.db"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")


class Base(DeclarativeBase):
    pass


class ArticleModel(Base):
    __tablename__ = "articles"

    news_id = Column(String(64), primary_key=True, index=True)
    title = Column(String(512), nullable=False)
    abstract = Column(Text, default="")
    category = Column(String(64), default="general", index=True)
    subcategory = Column(String(64), default="general")
    url = Column(String(1024), default="")
    persuasion_risk = Column(Float, default=0.0, index=True)
    persuasion_scores_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    @property
    def persuasion_scores(self) -> dict[str, float]:
        try:
            return json.loads(self.persuasion_scores_json or "{}")
        except Exception:
            return {}

    @persuasion_scores.setter
    def persuasion_scores(self, scores: dict[str, float]):
        self.persuasion_scores_json = json.dumps(scores)

    def to_dict(self) -> dict:
        return {
            "news_id": self.news_id,
            "title": self.title,
            "abstract": self.abstract,
            "category": self.category,
            "subcategory": self.subcategory,
            "url": self.url,
            "persuasion_risk": float(self.persuasion_risk or 0.0),
            "persuasion_scores": self.persuasion_scores,
        }


class UserModel(Base):
    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True, index=True)
    susceptibility = Column(Float, default=0.0)
    n_updates = Column(Integer, default=0)
    vector_json = Column(Text, default="[]")
    last_active = Column(DateTime, default=datetime.utcnow)

    @property
    def vector(self) -> list[float]:
        try:
            return json.loads(self.vector_json or "[]")
        except Exception:
            return [0.0] * 23

    @vector.setter
    def vector(self, vec: list[float]):
        self.vector_json = json.dumps(vec)

    def to_dict(self) -> dict:
        from src.models.persuasion_profile import TECHNIQUES
        v = self.vector
        top = [
            {"technique": TECHNIQUES[i], "score": float(v[i])}
            for i in range(min(len(TECHNIQUES), len(v)))
            if v[i] > 0.05
        ]
        top.sort(key=lambda x: -x["score"])
        return {
            "user_id": self.user_id,
            "susceptibility": float(self.susceptibility or 0.0),
            "n_updates": int(self.n_updates or 0),
            "vector": v,
            "top_techniques": top[:5],
        }


class InteractionModel(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), index=True, nullable=False)
    news_id = Column(String(64), index=True, nullable=False)
    action = Column(String(32), nullable=False)  # 'click' | 'skip' | 'flag'
    timestamp = Column(Float, default=time.time)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "news_id": self.news_id,
            "action": self.action,
            "timestamp": self.timestamp,
        }


# ── Database Initialization & Session Factory ────────────────────────────────

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables in the database if they do not exist."""
    Base.metadata.create_all(bind=engine)
    log.info("Database initialized at %s", DATABASE_URL)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
