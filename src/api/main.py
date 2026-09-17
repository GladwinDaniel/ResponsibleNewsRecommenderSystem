"""
main.py  —  FastAPI Backend for the Responsible News Recommender
----------------------------------------------------------------
Production-grade API providing:
  - Two-stage Responsible News Recommendation (FAISS retrieval + ethical reranking)
  - Sentence-level persuasion span highlighting
  - Closed-loop user feedback tracking with real-time profile adaptation
  - Pareto Frontier multi-objective research curves
  - Side-by-side comparator (Naive CTR vs. Responsible)
  - UN SDG 16 Governance & Audit Metrics
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ── Lifespan for Startup DB & Cache initialization ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up Responsible News Recommender API...")
    try:
        from src.data.init_db import seed_database
        seed_database()
        log.info("Database verified and seeded successfully.")
    except Exception as exc:
        log.warning("Database startup notice: %s", exc)
    yield
    log.info("Shutting down Responsible News Recommender API...")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Responsible News Recommender API",
    description=(
        "A persuasion-aware, knowledge-graph-augmented news recommender "
        "aligned with UN SDG 16: Peace, Justice and Strong Institutions."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Reranker Singleton ────────────────────────────────────────────────────────

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from src.ranking.reranker import get_reranker
        alpha        = float(os.getenv("ALPHA", "0.7"))
        beta         = float(os.getenv("BETA", "0.3"))
        provider_cap = float(os.getenv("PROVIDER_CAP", "0.30"))
        _reranker    = get_reranker(alpha=alpha, beta=beta, provider_cap=provider_cap)
    return _reranker


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ArticleSummary(BaseModel):
    news_id:            str
    title:              str
    abstract:           str = ""
    category:           str = ""
    url:                str = ""
    persuasion_risk:    float = 0.0
    risk_label:         str = "LOW"
    risk_emoji:         str = "🟢"
    risk_color:         str = "#22c55e"
    responsible_score:  float = 0.0
    relevance:          float = 0.0
    explanation:        str = ""
    top_techniques:     list[dict] = Field(default_factory=list)


class UserProfile(BaseModel):
    user_id:        str
    susceptibility: float = 0.0
    n_updates:      int   = 0
    vector:         list[float] = Field(default_factory=list)
    top_techniques: list[dict]  = Field(default_factory=list)


class ScoreRequest(BaseModel):
    text: str = Field(..., description="News article text to score for persuasion techniques")


class ScoreResponse(BaseModel):
    persuasion_risk:     float
    risk_label:          str
    risk_emoji:          str
    explanation:         str
    detected_techniques: list[dict]
    highlighted_spans:   list[dict] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    user_id: str
    news_id: str
    action:  Literal["click", "skip", "flag"] = Field(
        ..., description="'click' | 'skip' | 'flag'"
    )


class SDGMetrics(BaseModel):
    per:                  float
    provider_parity_gap:  float
    explainability_cover: float
    high_risk_articles:   int
    total_articles:       int
    avg_persuasion_risk:  float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health():
    """API liveness and status check."""
    return {
        "status":    "ok",
        "version":   "2.0.0",
        "timestamp": time.time(),
        "database":  "connected",
        "message":   "Responsible News Recommender API is healthy and running.",
    }


@app.get("/recommend/{user_id}", response_model=list[ArticleSummary], tags=["Recommend"])
async def recommend(
    user_id: str,
    top_k:   int   = Query(default=10, ge=1, le=100, description="Number of articles"),
    alpha:   float = Query(default=None, ge=0.0, le=1.0, description="Relevance weight"),
    beta:    float = Query(default=None, ge=0.0, le=1.0, description="Risk penalty weight"),
):
    """
    Return top-K responsibly-ranked news articles using two-stage retrieval.
    """
    try:
        ranker  = _get_reranker()
        results = ranker.rank(user_id=user_id, top_k=top_k, alpha=alpha, beta=beta)
        return [
            ArticleSummary(
                news_id          = r.get("news_id",           ""),
                title            = r.get("title",             ""),
                abstract         = r.get("abstract",          ""),
                category         = r.get("category",          ""),
                url              = r.get("url",               ""),
                persuasion_risk  = r.get("persuasion_risk",   0.0),
                risk_label       = r.get("risk_label",        "LOW"),
                risk_emoji       = r.get("risk_emoji",        "🟢"),
                risk_color       = r.get("risk_color",        "#22c55e"),
                responsible_score= r.get("responsible_score", 0.0),
                relevance        = r.get("relevance",         0.0),
                explanation      = r.get("explanation",       ""),
                top_techniques   = r.get("top_techniques",    []),
            )
            for r in results
        ]
    except Exception as exc:
        log.exception("Error in /recommend/%s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/recommend/{user_id}/compare", tags=["Research"])
async def compare_recommendations(user_id: str, top_k: int = Query(default=10, ge=1, le=50)):
    """
    Side-by-Side comparison between Naive CTR Recommendation and Responsible News Recommendation.
    Demonstrates Persuasion Exposure Reduction (PER) and high-risk suppression.
    """
    try:
        ranker = _get_reranker()
        return ranker.compare_rankings(user_id=user_id, top_k=top_k)
    except Exception as exc:
        log.exception("Error in /recommend/%s/compare: %s", user_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/article/{news_id}", tags=["Articles"])
async def get_article(news_id: str, user_id: str = Query(default="user_demo")):
    """
    Get full persuasion breakdown and sentence-level highlights for a specific article.
    """
    try:
        ranker = _get_reranker()
        result = ranker.score_article(user_id=user_id, news_id=news_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        from src.models.persuasion_detector import highlight_spans
        full_text = f"{result.get('title', '')}. {result.get('abstract', '')}"
        result["highlighted_spans"] = highlight_spans(full_text)

        return result
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Error in /article/%s: %s", news_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/user/{user_id}/profile", response_model=UserProfile, tags=["Users"])
async def get_user_profile(user_id: str):
    """
    Return a user's 23-technique persuasion profile from persistent storage.
    """
    try:
        ranker  = _get_reranker()
        profile = ranker.get_user_profile(user_id)
        return UserProfile(**{
            "user_id":        profile.get("user_id", user_id),
            "susceptibility": profile.get("susceptibility", 0.0),
            "n_updates":      profile.get("n_updates", 0),
            "vector":         profile.get("vector", [0.0] * 23),
            "top_techniques": profile.get("top_techniques", []),
        })
    except Exception as exc:
        log.exception("Error in /user/%s/profile: %s", user_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/score", response_model=ScoreResponse, tags=["Persuasion"])
async def score_text(body: ScoreRequest):
    """
    Score arbitrary news text with fine-grained sentence-level span highlighting.
    """
    try:
        from src.models.persuasion_detector import predict_text, highlight_spans, TECHNIQUES
        from src.ranking.responsible_score import risk_label, generate_explanation, TECHNIQUE_DESCRIPTIONS

        probs, risk = predict_text(body.text)
        label, emoji, _ = risk_label(risk)

        detected = [
            {
                "technique":   t,
                "probability": float(p),
                "description": TECHNIQUE_DESCRIPTIONS.get(t, ""),
            }
            for t, p in sorted(zip(TECHNIQUES, probs), key=lambda x: -x[1])
            if p > 0.3
        ]

        explanation = generate_explanation({
            "title":             body.text[:80],
            "persuasion_risk":   risk,
            "persuasion_scores": {t: float(p) for t, p in zip(TECHNIQUES, probs)},
        })

        spans = highlight_spans(body.text)

        return ScoreResponse(
            persuasion_risk     = risk,
            risk_label          = label,
            risk_emoji          = emoji,
            explanation         = explanation,
            detected_techniques = detected,
            highlighted_spans   = spans,
        )
    except Exception as exc:
        log.exception("Error in /score: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feedback", tags=["Feedback"])
async def record_feedback(body: FeedbackRequest):
    """
    Record explicit user feedback (click / skip / flag) in the database
    and dynamically update the user's EMA susceptibility profile.
    """
    try:
        from src.models.persuasion_profile import record_user_interaction
        updated_profile = record_user_interaction(
            user_id=body.user_id,
            news_id=body.news_id,
            action=body.action,
        )
        return {
            "status": "recorded",
            "user_id": body.user_id,
            "news_id": body.news_id,
            "action": body.action,
            "updated_profile": updated_profile,
        }
    except Exception as exc:
        log.exception("Error recording feedback: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/research/pareto", tags=["Research"])
async def pareto_frontier(user_id: str = Query(default="user_demo")):
    """
    Calculates the empirical multi-objective Pareto Frontier curve
    demonstrating the trade-off between user relevance and Persuasion Exposure Reduction.
    """
    try:
        from src.evaluation.pareto import compute_pareto_frontier
        return compute_pareto_frontier(user_id=user_id)
    except Exception as exc:
        log.exception("Error in /research/pareto: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics/sdg", response_model=SDGMetrics, tags=["SDG16"])
async def sdg_metrics():
    """
    System-wide SDG16 alignment metrics (PER, provider parity gap, explainability coverage).
    """
    try:
        ranker  = _get_reranker()
        metrics = ranker.sdg_metrics()
        return SDGMetrics(**{
            "per":                  metrics.get("per", 0.0),
            "provider_parity_gap":  metrics.get("provider_parity_gap", 0.0),
            "explainability_cover": metrics.get("explainability_cover", 0.0),
            "high_risk_articles":   metrics.get("high_risk_articles", 0),
            "total_articles":       metrics.get("total_articles", 0),
            "avg_persuasion_risk":  metrics.get("avg_persuasion_risk", 0.0),
        })
    except Exception as exc:
        log.exception("Error in /metrics/sdg: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/techniques", tags=["Meta"])
async def list_techniques():
    """Return the list of 23 persuasion techniques and their descriptions."""
    from src.ranking.responsible_score import TECHNIQUE_DESCRIPTIONS, DANGER_WEIGHTS, TECHNIQUES
    return [
        {
            "technique":    t,
            "description":  TECHNIQUE_DESCRIPTIONS.get(t, ""),
            "danger_weight": float(DANGER_WEIGHTS[i]),
        }
        for i, t in enumerate(TECHNIQUES)
    ]


@app.get("/demo", tags=["Meta"])
async def demo():
    """Quick demo endpoint."""
    ranker  = _get_reranker()
    results = ranker.rank(user_id="user_demo", top_k=5)
    sdg     = ranker.sdg_metrics()
    return {
        "message":        "Responsible News Recommender — Demo Mode",
        "recommendations": results[:5],
        "sdg_metrics":    sdg,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)
