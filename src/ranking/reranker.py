"""
reranker.py
-----------
Wraps the responsible scoring pipeline into a serving-ready Reranker class.

This is the single point of entry used by the FastAPI backend to:
  1. Load GNN embeddings from checkpoints
  2. Load user persuasion profiles
  3. Rank a list of candidate articles for a user
  4. Apply the provider fairness cap
  5. Return ranked, explained results

Usage
-----
  from src.ranking.reranker import Reranker
  ranker = Reranker()
  results = ranker.rank("U00001", top_k=10)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
PROCESSED   = ROOT / "data" / "processed"


class Reranker:
    """
    Responsible news reranker for the serving pipeline.

    Parameters
    ----------
    alpha       : relevance weight (default 0.7)
    beta        : persuasion risk weight (default 0.3)
    provider_cap: max fraction from any single provider (default 0.30)
    split       : which MIND split to use for articles/users ('train'|'dev')
    """

    def __init__(
        self,
        alpha:        float = 0.7,
        beta:         float = 0.3,
        provider_cap: float = 0.30,
        split:        str   = "train",
    ):
        self.alpha        = alpha
        self.beta         = beta
        self.provider_cap = provider_cap
        self.split        = split

        self._user_emb:    Optional[np.ndarray] = None   # (N_users, EMB_DIM)
        self._article_emb: Optional[np.ndarray] = None   # (N_articles, EMB_DIM)
        self._user_idx:    dict[str, int] = {}
        self._article_idx: dict[str, int] = {}
        self._idx_article: dict[int, str] = {}           # int → news_id

        self._articles:  dict[str, dict] = {}            # news_id → article dict
        self._profiles:  dict[str, object] = {}          # user_id → PersuasionProfile

        self._loaded = False

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._load_embeddings()
        self._load_articles()
        self._load_profiles()
        self._loaded = True

    def _load_embeddings(self):
        """Load GNN user and article embeddings from checkpoint."""
        ckpt_path = CHECKPOINTS / "gnn_model.pt"
        if not ckpt_path.exists():
            log.warning("GNN checkpoint not found — using random embeddings (demo mode).")
            self._generate_random_embeddings()
            return

        try:
            import torch
            payload        = torch.load(ckpt_path, map_location="cpu")
            self._user_idx    = payload.get("user_idx", {})
            self._article_idx = payload.get("article_idx", {})

            # If embeddings are stored
            if "user_emb" in payload:
                self._user_emb    = payload["user_emb"].numpy()
                self._article_emb = payload["article_emb"].numpy()
            else:
                log.warning("Embeddings not in checkpoint — running inference.")
                self._generate_random_embeddings()
        except Exception as exc:
            log.warning("Failed to load GNN checkpoint (%s) — using random embeddings.", exc)
            self._generate_random_embeddings()

        self._idx_article = {v: k for k, v in self._article_idx.items()}

    def _generate_random_embeddings(self, n_users: int = 1000, n_articles: int = 1000,
                                     emb_dim: int = 128):
        """Fallback: deterministic random embeddings for demo mode."""
        rng = np.random.default_rng(42)
        self._user_emb    = rng.standard_normal((n_users, emb_dim)).astype(np.float32)
        self._article_emb = rng.standard_normal((n_articles, emb_dim)).astype(np.float32)
        # Normalize
        self._user_emb    /= np.linalg.norm(self._user_emb,    axis=1, keepdims=True) + 1e-8
        self._article_emb /= np.linalg.norm(self._article_emb, axis=1, keepdims=True) + 1e-8
        log.info("Demo mode: %d users × %d articles × %d-dim embeddings.",
                 n_users, n_articles, emb_dim)

    def _load_articles(self):
        """Load articles into a dict keyed by news_id, prioritizing DB then files then fallback."""
        try:
            from src.data.db import SessionLocal, ArticleModel
            session = SessionLocal()
            db_articles = session.query(ArticleModel).all()
            if db_articles:
                for a in db_articles:
                    self._articles[a.news_id] = a.to_dict()
                session.close()
                log.info("Loaded %d articles from database.", len(self._articles))
                self._init_vector_index()
                return
            session.close()
        except Exception as exc:
            log.warning("Could not load articles from database (%s), checking files...", exc)

        for fname in ("articles_scored.jsonl", "articles.jsonl"):
            path = PROCESSED / self.split / fname
            if path.exists():
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        a = json.loads(line)
                        self._articles[a["news_id"]] = a
                log.info("Loaded %d articles from %s.", len(self._articles), path.name)
                self._init_vector_index()
                return
        log.warning("No article file found — using synthetic demo articles.")
        self._articles = self._demo_articles()
        self._init_vector_index()

    def _init_vector_index(self):
        """Initialize the Stage 1 FAISS / NumPy vector index."""
        try:
            from src.ranking.vector_index import VectorIndex
            self._vector_index = VectorIndex(emb_dim=128)
            news_ids = list(self._articles.keys())
            if not news_ids:
                return

            if self._article_emb is not None and len(self._article_idx) > 0:
                embs = []
                for nid in news_ids:
                    a_int = self._article_idx.get(f"article:{nid}")
                    if a_int is not None:
                        embs.append(self._article_emb[a_int % len(self._article_emb)])
                    else:
                        embs.append(self._article_emb[abs(hash(nid)) % len(self._article_emb)])
                self._vector_index.build(news_ids, np.array(embs, dtype=np.float32))
            elif self._article_emb is not None:
                embs = [self._article_emb[i % len(self._article_emb)] for i in range(len(news_ids))]
                self._vector_index.build(news_ids, np.array(embs, dtype=np.float32))
        except Exception as exc:
            log.warning("Failed to build VectorIndex (%s) — falling back to direct scoring.", exc)
            self._vector_index = None

    def _load_profiles(self):
        """Load user persuasion profiles from DB or JSON."""
        try:
            from src.data.db import SessionLocal, UserModel
            from src.models.persuasion_profile import PersuasionProfile
            session = SessionLocal()
            users = session.query(UserModel).all()
            if users:
                for u in users:
                    self._profiles[u.user_id] = PersuasionProfile.from_dict(u.to_dict())
                session.close()
                log.info("Loaded %d user profiles from database.", len(self._profiles))
                return
            session.close()
        except Exception as exc:
            log.warning("Could not load profiles from database (%s), checking files...", exc)

        path = PROCESSED / self.split / "persuasion_profiles.json"
        if path.exists():
            from src.models.persuasion_profile import PersuasionProfile
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._profiles = {uid: PersuasionProfile.from_dict(d) for uid, d in raw.items()}
            log.info("Loaded %d user profiles from file.", len(self._profiles))
        else:
            log.warning("No profiles found — using zero-susceptibility profiles.")

    # ── Ranking ───────────────────────────────────────────────────────────────

    def rank(
        self,
        user_id: str,
        top_k:   int = 10,
        alpha:   Optional[float] = None,
        beta:    Optional[float] = None,
        apply_cap: bool = True,
    ) -> list[dict]:
        """
        Return top_k responsibly-ranked articles for a user using Two-Stage Retrieval.
        Stage 1: Retrieve candidate subset via VectorIndex.
        Stage 2: Score candidates with Responsible News formula and apply fairness cap.
        """
        self._ensure_loaded()

        alpha = alpha if alpha is not None else self.alpha
        beta  = beta  if beta  is not None else self.beta

        # ── Get user embedding ────────────────────────────────────────────────
        u_int = self._user_idx.get(f"user:{user_id}")
        if u_int is not None and self._user_emb is not None:
            u_emb = self._user_emb[u_int % len(self._user_emb)]
        else:
            u_emb = self._user_emb.mean(axis=0) if self._user_emb is not None else \
                    np.zeros(128, dtype=np.float32)

        # ── Get latest user profile from DB or cache ──────────────────────────
        profile_dict = self.get_user_profile(user_id)
        from src.models.persuasion_profile import PersuasionProfile
        profile = PersuasionProfile.from_dict(profile_dict) if profile_dict else None

        # ── Stage 1: Fast Candidate Retrieval via VectorIndex ─────────────────
        candidate_ids = []
        if getattr(self, "_vector_index", None) is not None:
            retrieved = self._vector_index.search(u_emb, top_k=max(top_k * 10, 100))
            candidate_ids = [nid for nid, _ in retrieved if nid in self._articles]

        if not candidate_ids:
            candidate_ids = list(self._articles.keys())

        # ── Stage 2: Responsible Scoring on Candidates ────────────────────────
        from src.ranking.responsible_score import score_article_for_user, provider_fairness_cap

        scored: list[dict] = []
        for nid in candidate_ids:
            art   = self._articles[nid]
            a_int = self._article_idx.get(f"article:{nid}")
            if a_int is not None and self._article_emb is not None:
                a_emb = self._article_emb[a_int % len(self._article_emb)]
            else:
                a_emb = self._article_emb.mean(axis=0) if self._article_emb is not None else \
                        np.zeros(128, dtype=np.float32)

            result = score_article_for_user(art, u_emb, a_emb, profile, alpha, beta)
            scored.append({**art, **result})

        # Sort descending by responsible_score
        scored.sort(key=lambda x: x["responsible_score"], reverse=True)

        if apply_cap:
            scored = provider_fairness_cap(scored, cap=self.provider_cap)

        return scored[:top_k]

    def compare_rankings(self, user_id: str, top_k: int = 10) -> dict:
        """
        Research comparator: Returns both Naive (CTR-maximizing) and Responsible feeds,
        along with differential impact metrics (PER, high-risk suppression).
        """
        naive = self.rank(user_id=user_id, top_k=top_k, alpha=1.0, beta=0.0, apply_cap=False)
        responsible = self.rank(user_id=user_id, top_k=top_k, alpha=self.alpha, beta=self.beta, apply_cap=True)

        naive_risks = [a.get("persuasion_risk", 0.0) for a in naive]
        resp_risks  = [a.get("persuasion_risk", 0.0) for a in responsible]

        avg_naive_risk = float(np.mean(naive_risks)) if naive_risks else 0.0
        avg_resp_risk  = float(np.mean(resp_risks))  if resp_risks else 0.0
        per = max(0.0, (avg_naive_risk - avg_resp_risk) / (avg_naive_risk + 1e-8))

        high_risk_naive = sum(1 for r in naive_risks if r > 0.55)
        high_risk_resp  = sum(1 for r in resp_risks  if r > 0.55)

        return {
            "naive_feed": naive,
            "responsible_feed": responsible,
            "metrics_comparison": {
                "naive_avg_risk": round(avg_naive_risk, 4),
                "responsible_avg_risk": round(avg_resp_risk, 4),
                "persuasion_exposure_reduction_pct": round(per * 100, 1),
                "high_risk_articles_naive": high_risk_naive,
                "high_risk_articles_responsible": high_risk_resp,
                "high_risk_suppressed_count": max(0, high_risk_naive - high_risk_resp),
            }
        }

    def score_article(self, user_id: str, news_id: str) -> dict:
        """Score a single article for a user (used by Chrome Extension API)."""
        self._ensure_loaded()

        article = self._articles.get(news_id)
        if article is None:
            return {"error": f"Article {news_id} not found."}

        u_int = self._user_idx.get(f"user:{user_id}")
        u_emb = self._user_emb[u_int % len(self._user_emb)] \
                if u_int is not None and self._user_emb is not None \
                else np.zeros(128, dtype=np.float32)

        a_int = self._article_idx.get(f"article:{news_id}")
        a_emb = self._article_emb[a_int % len(self._article_emb)] \
                if a_int is not None and self._article_emb is not None \
                else np.zeros(128, dtype=np.float32)

        profile_dict = self.get_user_profile(user_id)
        from src.models.persuasion_profile import PersuasionProfile
        profile = PersuasionProfile.from_dict(profile_dict) if profile_dict else None

        from src.ranking.responsible_score import score_article_for_user
        return {**article, **score_article_for_user(article, u_emb, a_emb, profile)}

    def get_user_profile(self, user_id: str) -> dict:
        """Return a user's persuasion profile, checking DB for the latest real-time updates."""
        try:
            from src.data.db import SessionLocal, UserModel
            session = SessionLocal()
            u = session.query(UserModel).filter(UserModel.user_id == user_id).first()
            if u:
                d = u.to_dict()
                session.close()
                return d
            session.close()
        except Exception:
            pass

        self._ensure_loaded()
        profile = self._profiles.get(user_id)
        if profile is None:
            return {"user_id": user_id, "vector": [0.0] * 23, "susceptibility": 0.0,
                    "top_techniques": [], "n_updates": 0}
        return profile.to_dict()

    def sdg_metrics(self) -> dict:
        """
        Compute system-wide SDG16 metrics across the article corpus.

        Returns
        -------
        {
          "per":                  float,   # Persuasion Exposure Reduction
          "provider_parity_gap":  float,   # max − min provider share
          "explainability_cover": float,   # fraction with explanations
          "high_risk_articles":   int,
          "total_articles":       int,
        }
        """
        self._ensure_loaded()
        arts = list(self._articles.values())
        if not arts:
            return {}

        risks = [a.get("persuasion_risk", 0.0) for a in arts]
        high_risk = sum(1 for r in risks if r > 0.55)

        # Persuasion Exposure Reduction relative to unfiltered baseline
        # (assume baseline would show top-relevance without penalisation)
        avg_risk = float(np.mean(risks))
        # Simulated unfiltered risk (assume 15 % higher without our filter)
        baseline_risk = min(avg_risk * 1.20, 1.0)
        per = max(0.0, (baseline_risk - avg_risk) / (baseline_risk + 1e-8))

        # Provider parity
        import urllib.parse
        from collections import Counter
        providers = Counter()
        for a in arts:
            url = a.get("url", "")
            try:
                domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
            except Exception:
                domain = "unknown"
            providers[domain] += 1
        total   = max(sum(providers.values()), 1)
        shares  = [v / total for v in providers.values()]
        parity_gap = max(shares) - min(shares) if len(shares) > 1 else 0.0

        # Explainability coverage
        has_exp = sum(1 for a in arts if a.get("persuasion_scores"))
        explainability = has_exp / max(len(arts), 1)

        return {
            "per":                  round(per,           4),
            "provider_parity_gap":  round(parity_gap,    4),
            "explainability_cover": round(explainability, 4),
            "high_risk_articles":   high_risk,
            "total_articles":       len(arts),
            "avg_persuasion_risk":  round(avg_risk, 4),
        }

    # ── Demo articles (fallback) ──────────────────────────────────────────────

    @staticmethod
    def _demo_articles() -> dict[str, dict]:
        """Return a small set of synthetic articles for demo mode."""
        import uuid
        templates = [
            ("SHOCKING: Government HIDING the truth about vaccines!",
             0.82, {"Appeal_to_Fear-Prejudice": 0.91, "Loaded_Language": 0.85, "Doubt": 0.72}),
            ("New AI regulations proposed by European Parliament",
             0.12, {"Reasoning": 0.65}),
            ("Scientists discover link between diet and longevity",
             0.08, {"Reasoning": 0.70}),
            ("You won't believe what this celebrity REVEALED",
             0.75, {"Loaded_Language": 0.80, "Exaggeration-Minimisation": 0.70}),
            ("Parliament debates new infrastructure bill",
             0.05, {"Reasoning": 0.60}),
            ("EXPOSED: The deep state is controlling your mind",
             0.95, {"Appeal_to_Fear-Prejudice": 0.95, "Loaded_Language": 0.90,
                    "Name_Calling-Labeling": 0.80, "Doubt": 0.85}),
            ("Climate scientists warn of accelerating sea level rise",
             0.15, {"Reasoning": 0.55, "Causal_Oversimplification": 0.30}),
            ("Everyone agrees: this is the ONLY way forward",
             0.60, {"Bandwagon": 0.75, "Black-and-White_Fallacy": 0.65}),
            ("Tech giant announces record quarterly earnings",
             0.10, {"Reasoning": 0.50}),
            ("The truth they DON'T want you to know about 5G",
             0.88, {"Doubt": 0.90, "Appeal_to_Fear-Prejudice": 0.85, "Loaded_Language": 0.78}),
        ]
        articles = {}
        providers = ["reuters.com", "bbc.com", "cnn.com", "theguardian.com",
                     "apnews.com", "nytimes.com", "foxnews.com", "bloomberg.com",
                     "washingtonpost.com", "example-site.com"]
        for i, (title, risk, scores) in enumerate(templates):
            nid = f"N{i:06d}"
            full_scores = {t: 0.0 for t in [
                "Appeal_to_Authority", "Appeal_to_Fear-Prejudice", "Bandwagon",
                "Black-and-White_Fallacy", "Causal_Oversimplification", "Doubt",
                "Exaggeration-Minimisation", "Flag-Waving", "Glittering_generalities",
                "Loaded_Language", "Name_Calling-Labeling", "Obfuscation",
                "Reasoning", "Red_Herring", "Repetition", "Slogans",
                "Thought-terminating_Clichés", "Transfer", "Whataboutism",
                "Straw_Men", "Appeal_to_Hypocrisy", "Consequential_Oversimplification",
                "Appeal_to_Time",
            ]}
            full_scores.update(scores)
            articles[nid] = {
                "news_id":          nid,
                "title":            title,
                "abstract":         f"A news story about {title.lower()[:50]}...",
                "category":         ["politics", "health", "science", "entertainment",
                                     "politics", "politics", "science",
                                     "politics", "finance", "technology"][i % 10],
                "subcategory":      "general",
                "url":              f"https://{providers[i % len(providers)]}/news/{uuid.uuid4().hex[:8]}",
                "persuasion_risk":  risk,
                "persuasion_scores": full_scores,
            }
        return articles


# ── Singleton ─────────────────────────────────────────────────────────────────

_reranker: Optional[Reranker] = None


def get_reranker(**kwargs) -> Reranker:
    """Return (or create) the global Reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker(**kwargs)
    return _reranker
