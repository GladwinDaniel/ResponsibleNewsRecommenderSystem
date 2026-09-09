"""
pareto.py  —  Pareto Frontier Trade-Off Analysis for Responsible News Recommendation
-----------------------------------------------------------------------------------
Scientifically evaluates the trade-off curve between:
  1. Content Relevance / Engagement (GNN similarity)
  2. Persuasion Exposure Reduction (PER)
  3. Provider Source Diversity (Parity Gap)

Used by researchers and auditors to find the optimal operating point (default α=0.7, β=0.3).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def compute_pareto_frontier(
    user_id: str = "user_demo",
    steps: int = 11,
    top_k: int = 10,
) -> list[dict]:
    """
    Sweep beta from 0.0 (unfiltered CTR optimization) to 1.0 (maximum defensive filtering),
    recording the multi-objective Pareto metrics for each point.

    Returns
    -------
    List of dicts:
      [
        {
          "beta": float,
          "alpha": float,
          "avg_relevance": float,
          "avg_persuasion_risk": float,
          "persuasion_exposure_reduction_pct": float,
          "high_risk_count": int,
          "provider_diversity_gap": float,
          "is_recommended_operating_point": bool,
        },
        ...
      ]
    """
    from src.ranking.reranker import get_reranker
    ranker = get_reranker()

    betas = np.linspace(0.0, 1.0, steps)
    frontier_points = []

    # Get baseline unfiltered risk (beta=0)
    baseline_recs = ranker.rank(user_id=user_id, top_k=top_k, alpha=1.0, beta=0.0, apply_cap=False)
    baseline_risks = [a.get("persuasion_risk", 0.0) for a in baseline_recs]
    baseline_avg_risk = float(np.mean(baseline_risks)) if baseline_risks else 0.5

    for b in betas:
        b_val = round(float(b), 2)
        a_val = round(float(1.0 - 0.5 * b_val), 2)

        recs = ranker.rank(user_id=user_id, top_k=top_k, alpha=a_val, beta=b_val, apply_cap=True)

        relevances = [a.get("relevance", 0.0) for a in recs]
        risks = [a.get("persuasion_risk", 0.0) for a in recs]

        avg_rel = float(np.mean(relevances)) if relevances else 0.0
        avg_risk = float(np.mean(risks)) if risks else 0.0

        per = max(0.0, (baseline_avg_risk - avg_risk) / (baseline_avg_risk + 1e-8))
        high_risk_count = sum(1 for r in risks if r > 0.55)

        # Provider diversity gap
        import urllib.parse
        from collections import Counter
        counts = Counter()
        for a in recs:
            try:
                counts[urllib.parse.urlparse(a.get("url", "")).netloc.replace("www.", "")] += 1
            except Exception:
                counts["unknown"] += 1
        shares = [c / len(recs) for c in counts.values()] if recs else [0.0]
        parity_gap = max(shares) - min(shares) if len(shares) > 1 else 0.0

        is_optimal = abs(b_val - 0.3) < 0.05

        frontier_points.append({
            "beta": b_val,
            "alpha": a_val,
            "avg_relevance": round(avg_rel, 3),
            "avg_persuasion_risk": round(avg_risk, 3),
            "persuasion_exposure_reduction_pct": round(per * 100, 1),
            "high_risk_count": high_risk_count,
            "provider_diversity_gap": round(parity_gap, 3),
            "is_recommended_operating_point": is_optimal,
        })

    return frontier_points
