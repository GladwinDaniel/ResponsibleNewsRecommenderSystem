"""
persuasion_profile.py
---------------------
Builds and updates per-user persuasion susceptibility profiles using
Exponential Moving Average (EMA) over the persuasion scores of articles
the user has clicked.

A profile is a 23-dimensional vector (one weight per technique).
High values indicate the user frequently engages with content using that
technique — i.e. they are *susceptible* to it.

Usage
-----
  python src/models/persuasion_profile.py  [--split train|dev]
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

TECHNIQUES = [
    "Appeal_to_Authority",
    "Appeal_to_Fear-Prejudice",
    "Bandwagon",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration-Minimisation",
    "Flag-Waving",
    "Glittering_generalities",
    "Loaded_Language",
    "Name_Calling-Labeling",
    "Obfuscation",
    "Reasoning",
    "Red_Herring",
    "Repetition",
    "Slogans",
    "Thought-terminating_Clichés",
    "Transfer",
    "Whataboutism",
    "Straw_Men",
    "Appeal_to_Hypocrisy",
    "Consequential_Oversimplification",
    "Appeal_to_Time",
]
NUM_TECHNIQUES = len(TECHNIQUES)


# ── EMA Profile class ─────────────────────────────────────────────────────────

class PersuasionProfile:
    """
    Per-user EMA-based susceptibility profile.

    alpha controls how quickly the profile responds to new signal:
      - alpha=0.1 → slow, stable  (long-term reading habits)
      - alpha=0.3 → medium        (recommended default)
      - alpha=0.7 → fast, reactive
    """

    def __init__(self, user_id: str, alpha: float = 0.3):
        self.user_id    = user_id
        self.alpha      = alpha
        self.vector     = np.zeros(NUM_TECHNIQUES, dtype=np.float32)
        self.n_updates  = 0
        self.initialized = False

    def update(self, persuasion_scores: dict[str, float]):
        """Update profile with scores from a newly clicked article."""
        self.update_with_action(persuasion_scores, action="click")

    def update_with_action(self, persuasion_scores: dict[str, float], action: str = "click"):
        """
        Update profile according to user interaction type:
          - 'click': User willingly engaged; standard EMA adaptation.
          - 'flag': User flagged manipulative content; raises defensive susceptibility weight so
                    similar techniques receive maximum suppression in future rerankings.
          - 'skip': User ignored; slight decay to reflect lack of engagement.
        """
        vec = np.array([persuasion_scores.get(t, 0.0) for t in TECHNIQUES],
                       dtype=np.float32)
        if action == "flag":
            # Strongly increase sensitivity/shielding for the flagged techniques
            self.vector = np.clip(self.vector + 0.25 * vec, 0.0, 1.0)
            self.initialized = True
        elif action == "skip":
            self.vector = np.clip(self.vector * 0.98, 0.0, 1.0)
        else:  # 'click'
            if not self.initialized:
                self.vector = vec
                self.initialized = True
            else:
                self.vector = (1 - self.alpha) * self.vector + self.alpha * vec
        self.n_updates += 1

    def update_from_article(self, article: dict, action: str = "click"):
        """Convenience wrapper accepting a processed article dict."""
        scores = article.get("persuasion_scores", {})
        if scores:
            self.update_with_action(scores, action=action)

    def susceptibility(self) -> float:
        """
        Overall susceptibility score in [0, 1]:
          weighted sum of profile vector (high-risk techniques weighted more).
        """
        from src.models.persuasion_detector import DANGER_WEIGHTS  # local import
        dw = DANGER_WEIGHTS[:NUM_TECHNIQUES]
        return float(np.dot(self.vector, dw) / (dw.sum() + 1e-8))

    def to_dict(self) -> dict:
        return {
            "user_id":     self.user_id,
            "alpha":       self.alpha,
            "n_updates":   self.n_updates,
            "vector":      self.vector.tolist(),
            "susceptibility": self.susceptibility(),
            "top_techniques": self._top_techniques(k=5),
        }

    def _top_techniques(self, k: int = 5) -> list[dict]:
        idx_sorted = np.argsort(self.vector)[::-1][:k]
        return [
            {"technique": TECHNIQUES[i], "score": float(self.vector[i])}
            for i in idx_sorted if self.vector[i] > 0.05
        ]

    @classmethod
    def from_dict(cls, data: dict) -> "PersuasionProfile":
        p = cls(data["user_id"], data.get("alpha", 0.3))
        p.vector      = np.array(data.get("vector", [0.0] * NUM_TECHNIQUES), dtype=np.float32)
        p.n_updates   = data.get("n_updates", 0)
        p.initialized = True
        return p


# ── Database Interaction Helper ───────────────────────────────────────────────

def record_user_interaction(user_id: str, news_id: str, action: str) -> dict:
    """
    Persist an interaction to DB and update the user's persuasion profile in real time.
    """
    from src.data.db import SessionLocal, UserModel, ArticleModel, InteractionModel
    session = SessionLocal()
    try:
        # 1. Record interaction
        interaction = InteractionModel(
            user_id=user_id,
            news_id=news_id,
            action=action,
        )
        session.add(interaction)

        # 2. Fetch article scores
        article = session.query(ArticleModel).filter(ArticleModel.news_id == news_id).first()
        scores = article.persuasion_scores if article else {}

        # 3. Fetch or create user
        user_row = session.query(UserModel).filter(UserModel.user_id == user_id).first()
        if user_row is None:
            user_row = UserModel(
                user_id=user_id,
                susceptibility=0.4,
                n_updates=0,
                vector_json=json.dumps([0.0] * NUM_TECHNIQUES),
            )
            session.add(user_row)
            session.flush()

        # 4. Update profile using EMA
        profile = PersuasionProfile.from_dict(user_row.to_dict())
        profile.update_with_action(scores, action=action)

        user_row.vector = profile.vector.tolist()
        user_row.susceptibility = profile.susceptibility()
        user_row.n_updates = profile.n_updates

        session.commit()
        return profile.to_dict()
    except Exception as exc:
        session.rollback()
        log.exception("Failed to record interaction in DB: %s", exc)
        return {"user_id": user_id, "error": str(exc)}
    finally:
        session.close()


# ── Batch profile builder ─────────────────────────────────────────────────────

def build_profiles(split: str = "train", alpha: float = 0.3) -> dict[str, dict]:
    """
    Construct persuasion profiles for all users in a MIND split.

    Algorithm:
      1. Load users.json (user_id → click history list)
      2. Load articles_scored.jsonl (news_id → persuasion_scores)
      3. For each user, EMA-update profile over their click history
      4. Write profiles to persuasion_profiles.json
    """
    split_dir = PROCESSED / split
    users_path    = split_dir / "users.json"
    articles_path = split_dir / "articles_scored.jsonl"
    out_path      = split_dir / "persuasion_profiles.json"

    # ── Load article scores ───────────────────────────────────────────────────
    if not articles_path.exists():
        log.warning("articles_scored.jsonl not found — using zero persuasion scores.")
        article_db: dict[str, dict] = {}
    else:
        log.info("Loading article scores from %s …", articles_path)
        article_db = {}
        with open(articles_path, encoding="utf-8") as fh:
            for line in fh:
                a = json.loads(line)
                article_db[a["news_id"]] = a

    # ── Load users ────────────────────────────────────────────────────────────
    if not users_path.exists():
        log.warning("users.json not found — no profiles to build.")
        return {}

    with open(users_path, encoding="utf-8") as fh:
        users = json.load(fh)
    log.info("Building profiles for %d users (alpha=%.2f) …", len(users), alpha)

    profiles: dict[str, dict] = {}
    for user_id, data in users.items():
        p = PersuasionProfile(user_id, alpha=alpha)
        history = data.get("history", []) + data.get("impressions", [])
        for news_id in history:
            article = article_db.get(news_id)
            if article and article.get("persuasion_scores"):
                p.update_from_article(article)
            else:
                # article not scored yet → small uniform push (prior)
                p.update({t: 0.05 for t in TECHNIQUES})
        profiles[user_id] = p.to_dict()

    # Save
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(profiles, fh, indent=2)
    log.info("✅  Profiles written → %s", out_path)
    return profiles


def load_profiles(split: str = "train") -> dict[str, "PersuasionProfile"]:
    """Load saved profiles back as PersuasionProfile objects."""
    path = PROCESSED / split / "persuasion_profiles.json"
    if not path.exists():
        log.warning("No profiles found at %s.", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {uid: PersuasionProfile.from_dict(data) for uid, data in raw.items()}


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    """Print a sample profile built on synthetic click sequence."""
    scores_low  = {t: 0.05 for t in TECHNIQUES}
    scores_high = {t: 0.8 for t in ["Loaded_Language", "Appeal_to_Fear-Prejudice",
                                     "Name_Calling-Labeling"]}
    scores_high.update({t: 0.05 for t in TECHNIQUES if t not in scores_high})

    p = PersuasionProfile("demo_user_001", alpha=0.3)
    for _ in range(5):
        p.update(scores_low)
    for _ in range(15):
        p.update(scores_high)

    data = p.to_dict()
    print(f"\n👤  Profile for {data['user_id']}")
    print(f"   Overall susceptibility: {data['susceptibility']:.3f}")
    print(f"   Updates: {data['n_updates']}")
    print("   Top techniques:")
    for item in data["top_techniques"]:
        bar = "█" * int(item["score"] * 20)
        print(f"     {item['technique']:<42} {item['score']:.3f}  {bar}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default="train", choices=["train", "dev"])
    parser.add_argument("--alpha",  type=float, default=0.3)
    parser.add_argument("--demo",   action="store_true")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        build_profiles(split=args.split, alpha=args.alpha)
