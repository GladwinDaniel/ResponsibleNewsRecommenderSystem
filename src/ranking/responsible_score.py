"""
responsible_score.py
--------------------
Computes the composite responsible recommendation score:

    Score(u, n) = α · Relevance(u, n)  −  β · PersuasionRisk(n, u)
                + γ · Explainability(n)

Where:
  Relevance(u, n)    = dot(user_emb[u], article_emb[n])  (from GNN)
  PersuasionRisk(n,u)= weighted combo of article's technique scores
                       boosted by user's susceptibility to those techniques
  Explainability(n)  = 1 if the article has a persuasion explanation, else 0
  α, β, γ            = tunable weights (default 0.7, 0.3, 0.05)

Also provides:
  - `generate_explanation(article, user_profile)` → human-readable string
  - `provider_fairness_cap(candidates, cap)` → enforces source diversity
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

TECHNIQUES = [
    "Appeal_to_Authority", "Appeal_to_Fear-Prejudice", "Bandwagon",
    "Black-and-White_Fallacy", "Causal_Oversimplification", "Doubt",
    "Exaggeration-Minimisation", "Flag-Waving", "Glittering_generalities",
    "Loaded_Language", "Name_Calling-Labeling", "Obfuscation",
    "Reasoning", "Red_Herring", "Repetition", "Slogans",
    "Thought-terminating_Clichés", "Transfer", "Whataboutism",
    "Straw_Men", "Appeal_to_Hypocrisy", "Consequential_Oversimplification",
    "Appeal_to_Time",
]

DANGER_WEIGHTS = np.array([
    0.6, 0.9, 0.7, 0.8, 0.7, 0.8, 0.6, 0.7, 0.5, 0.9,
    0.85, 0.6, 0.3, 0.5, 0.4, 0.5, 0.7, 0.6, 0.7, 0.8,
    0.75, 0.6, 0.3,
], dtype=np.float32)

# Human-readable descriptions for each technique
TECHNIQUE_DESCRIPTIONS = {
    "Appeal_to_Authority":           "Citing an authority figure without critical evaluation",
    "Appeal_to_Fear-Prejudice":      "Exploiting fear or prejudice to manipulate opinion",
    "Bandwagon":                     "Claiming everyone believes something to pressure agreement",
    "Black-and-White_Fallacy":       "Presenting only two extreme options, ignoring nuance",
    "Causal_Oversimplification":     "Attributing a complex effect to a single cause",
    "Doubt":                         "Casting unjustified doubt on credible sources",
    "Exaggeration-Minimisation":     "Inflating or deflating the significance of events",
    "Flag-Waving":                   "Using national or group identity to justify a position",
    "Glittering_generalities":       "Using vague, feel-good words to gain approval",
    "Loaded_Language":               "Using emotionally charged words to sway opinion",
    "Name_Calling-Labeling":         "Attaching a negative label to dismiss an argument",
    "Obfuscation":                   "Deliberately making content confusing or unclear",
    "Reasoning":                     "Providing logical, evidence-based argument (constructive)",
    "Red_Herring":                   "Introducing an irrelevant topic to distract",
    "Repetition":                    "Repeating a claim to make it seem more true",
    "Slogans":                       "Using catchy phrases as substitutes for argument",
    "Thought-terminating_Clichés":   "Using clichés to stop critical thinking",
    "Transfer":                      "Associating something with a respected/disliked symbol",
    "Whataboutism":                  "Deflecting criticism by pointing to others' faults",
    "Straw_Men":                     "Misrepresenting opponent's argument to attack it easily",
    "Appeal_to_Hypocrisy":           "Dismissing criticism by pointing to the critic's flaws",
    "Consequential_Oversimplification": "Oversimplifying consequences to support a conclusion",
    "Appeal_to_Time":                "Claiming something is right because it is old or new",
}

# Risk level thresholds
RISK_LABELS = {
    (0.0,  0.25): ("LOW",    "🟢", "#22c55e"),
    (0.25, 0.55): ("MEDIUM", "🟡", "#f59e0b"),
    (0.55, 1.01): ("HIGH",   "🔴", "#ef4444"),
}


def risk_label(risk: float) -> tuple[str, str, str]:
    """Return (label, emoji, hex_color) for a risk score."""
    for (lo, hi), result in RISK_LABELS.items():
        if lo <= risk < hi:
            return result
    return ("HIGH", "🔴", "#ef4444")


# ── Core scoring function ─────────────────────────────────────────────────────

def responsible_score(
    relevance:          float,
    article_risk:       float,
    user_susceptibility: float = 0.5,
    alpha: float = 0.7,
    beta:  float = 0.3,
    gamma: float = 0.05,
    has_explanation: bool = True,
) -> float:
    """
    Compute the responsible recommendation score in [0, 1].

    Parameters
    ----------
    relevance           : raw GNN relevance score (0–1)
    article_risk        : article persuasion risk (0–1)
    user_susceptibility : user's susceptibility to persuasion (0–1)
    alpha               : weight for relevance
    beta                : weight for risk penalty
    gamma               : bonus for articles with explanations
    has_explanation     : whether the article has a generated explanation
    """
    # Personalised risk = article risk amplified by user susceptibility
    personalised_risk = article_risk * (0.5 + 0.5 * user_susceptibility)
    explainability    = gamma if has_explanation else 0.0
    raw = alpha * relevance - beta * personalised_risk + explainability

    # Clamp to [0, 1]
    return float(np.clip(raw, 0.0, 1.0))


def score_article_for_user(
    article:        dict,
    user_emb:       "np.ndarray",
    article_emb:    "np.ndarray",
    user_profile:   Optional["PersuasionProfile"] = None,  # type: ignore
    alpha: float = 0.7,
    beta:  float = 0.3,
) -> dict:
    """
    Compute the full score dict for one (user, article) pair.

    Returns
    -------
    {
      "responsible_score": float,
      "relevance":         float,
      "persuasion_risk":   float,
      "susceptibility":    float,
      "risk_label":        str,
      "risk_emoji":        str,
      "risk_color":        str,
      "explanation":       str,
      "top_techniques":    list[dict],
    }
    """
    # Relevance from GNN dot product
    relevance = float(np.clip(np.dot(user_emb, article_emb) /
                              (np.linalg.norm(user_emb) * np.linalg.norm(article_emb) + 1e-8),
                              0.0, 1.0))

    article_risk  = article.get("persuasion_risk", 0.0)
    susceptibility = user_profile.susceptibility() if user_profile else 0.5
    score          = responsible_score(relevance, article_risk, susceptibility, alpha, beta)

    label, emoji, color = risk_label(article_risk)
    explanation = generate_explanation(article, user_profile)

    top_techs = sorted(
        [{"technique": t, "score": float(article.get("persuasion_scores", {}).get(t, 0.0))}
         for t in TECHNIQUES],
        key=lambda x: -x["score"],
    )[:5]

    return {
        "responsible_score": score,
        "relevance":         relevance,
        "persuasion_risk":   article_risk,
        "susceptibility":    susceptibility,
        "risk_label":        label,
        "risk_emoji":        emoji,
        "risk_color":        color,
        "explanation":       explanation,
        "top_techniques":    top_techs,
    }


# ── Explanation generator ─────────────────────────────────────────────────────

def generate_explanation(article: dict, user_profile=None) -> str:
    """
    Produce a human-readable explanation of why the article was ranked as it was.

    Returns a plain-English string suitable for display in the UI.
    """
    scores = article.get("persuasion_scores", {})
    risk   = article.get("persuasion_risk", 0.0)
    title  = article.get("title", "This article")
    label, emoji, _ = risk_label(risk)

    detected = sorted(
        [(t, scores.get(t, 0.0)) for t in TECHNIQUES if scores.get(t, 0.0) > 0.3],
        key=lambda x: -x[1],
    )[:3]

    if risk < 0.25:
        base = f"{emoji} This article uses minimal persuasion techniques and appears informational."
    elif risk < 0.55:
        base = f"{emoji} This article uses moderate persuasion techniques."
    else:
        base = f"{emoji} This article uses strong persuasion techniques."

    if detected:
        desc_list = "; ".join(
            f"**{t.replace('_', ' ')}** ({TECHNIQUE_DESCRIPTIONS.get(t, '')})"
            for t, _ in detected
        )
        base += f" Detected: {desc_list}."

    if user_profile:
        susc = user_profile.susceptibility()
        if susc > 0.6:
            base += (
                " ⚠️ Based on your reading history, you appear susceptible to persuasion "
                "techniques in this article. Consider verifying claims independently."
            )
        elif susc < 0.2:
            base += " Your reading history suggests you are relatively resistant to these techniques."

    return base


# ── Provider fairness cap ─────────────────────────────────────────────────────

def provider_fairness_cap(
    candidates: list[dict],
    cap: float = 0.30,
    provider_field: str = "url",
) -> list[dict]:
    """
    Enforce source diversity by limiting any single provider to at most
    `cap` fraction of the final recommendation list.

    Assumes each article dict has a 'url' field; provider is extracted
    from the domain.
    """
    import urllib.parse

    def _provider(article: dict) -> str:
        url = article.get(provider_field, "")
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc.replace("www.", "")
        except Exception:
            return "unknown"

    if not candidates:
        return candidates

    max_per_provider = max(1, int(len(candidates) * cap))
    filtered: list[dict]    = []
    provider_counts: dict[str, int] = {}

    for art in candidates:
        prov = _provider(art)
        count = provider_counts.get(prov, 0)
        if count < max_per_provider:
            filtered.append(art)
            provider_counts[prov] = count + 1

    return filtered


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    test_articles = [
        {
            "news_id":  "N000001",
            "title":    "SHOCKING: Government HIDING the truth about vaccines!",
            "url":      "https://example1.com/news/abc",
            "persuasion_risk": 0.82,
            "persuasion_scores": {
                "Appeal_to_Fear-Prejudice": 0.91, "Loaded_Language": 0.85,
                "Doubt": 0.72, "Name_Calling-Labeling": 0.60,
            },
        },
        {
            "news_id":  "N000002",
            "title":    "Scientists publish new findings on climate resilience",
            "url":      "https://reuters.com/news/def",
            "persuasion_risk": 0.08,
            "persuasion_scores": {"Reasoning": 0.72},
        },
        {
            "news_id":  "N000003",
            "title":    "Our nation stands united — only together can we win",
            "url":      "https://example1.com/news/ghi",
            "persuasion_risk": 0.48,
            "persuasion_scores": {"Flag-Waving": 0.65, "Bandwagon": 0.55},
        },
    ]

    np.random.seed(42)
    user_emb    = np.random.randn(128).astype(np.float32)
    article_embs = [np.random.randn(128).astype(np.float32) for _ in test_articles]

    print("\n📊  Responsible Score Demo\n" + "=" * 60)
    for art, a_emb in zip(test_articles, article_embs):
        result = score_article_for_user(art, user_emb, a_emb)
        print(f"\n  {result['risk_emoji']} [{result['risk_label']}] {art['title'][:60]}")
        print(f"     responsible_score = {result['responsible_score']:.3f} "
              f"| relevance = {result['relevance']:.3f} "
              f"| risk = {result['persuasion_risk']:.3f}")
        print(f"     Explanation: {result['explanation'][:120]}")

    print("\n\n🎯  Provider Fairness Cap (cap=33%)")
    capped = provider_fairness_cap(test_articles, cap=0.33)
    for art in capped:
        print(f"  ✅ {art['title'][:60]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
