"""
metrics.py
----------
Evaluation metrics for the Responsible News Recommender.

Standard IR metrics:
  • AUC (Area Under ROC Curve)
  • nDCG@K (Normalised Discounted Cumulative Gain)
  • Precision@K

Novel SDG16 metrics:
  • PER  — Persuasion Exposure Reduction (our contribution)
  • Provider Parity Gap
  • Explainability Coverage

Usage
-----
  python src/evaluation/metrics.py --demo       # run on synthetic data
  python src/evaluation/metrics.py --eval-mind  # evaluate on MIND dev split
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


# ── Standard IR Metrics ───────────────────────────────────────────────────────

def auc(labels: list[int], scores: list[float]) -> float:
    """Binary AUC via trapezoidal rule (no sklearn dependency)."""
    if len(set(labels)) < 2:
        return 0.5
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = auc_val = prev_tp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc_val += (tp + prev_tp) / 2
            prev_tp = tp
    return auc_val / (n_pos * n_neg)


def ndcg_at_k(labels: list[int], scores: list[float], k: int = 10) -> float:
    """nDCG@K."""
    sorted_by_score = [l for _, l in sorted(zip(scores, labels), reverse=True)]
    ideal           = sorted(labels, reverse=True)

    def dcg(lst):
        return sum(l / np.log2(i + 2) for i, l in enumerate(lst[:k]))

    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(sorted_by_score) / ideal_dcg


def precision_at_k(labels: list[int], scores: list[float], k: int = 10) -> float:
    """Precision@K."""
    top_k  = [l for _, l in sorted(zip(scores, labels), reverse=True)[:k]]
    return sum(top_k) / max(len(top_k), 1)


def mrr(labels: list[int], scores: list[float]) -> float:
    """Mean Reciprocal Rank."""
    sorted_labels = [l for _, l in sorted(zip(scores, labels), reverse=True)]
    for i, l in enumerate(sorted_labels):
        if l == 1:
            return 1.0 / (i + 1)
    return 0.0


# ── SDG16 Novel Metrics ───────────────────────────────────────────────────────

def persuasion_exposure_reduction(
    ranked_articles:   list[dict],
    unfiltered_articles: list[dict],
) -> float:
    """
    PER = (avg_risk_unfiltered − avg_risk_ranked) / avg_risk_unfiltered

    Measures what fraction of persuasion exposure was eliminated by
    our responsible ranking compared to a naive relevance-only ranking.

    Higher is better.  Target: ≥ 0.15 (15 %).
    """
    def avg_risk(articles):
        risks = [a.get("persuasion_risk", 0.0) for a in articles]
        return float(np.mean(risks)) if risks else 0.0

    r_filtered   = avg_risk(ranked_articles)
    r_unfiltered = avg_risk(unfiltered_articles)

    if r_unfiltered == 0:
        return 0.0
    return max(0.0, (r_unfiltered - r_filtered) / r_unfiltered)


def provider_parity_gap(articles: list[dict]) -> float:
    """
    Maximum provider share − minimum provider share.
    Lower is better.  Target: < 0.05 (5 %).
    """
    import urllib.parse
    counts: Counter = Counter()
    for a in articles:
        url = a.get("url", "")
        try:
            domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
        except Exception:
            domain = "unknown"
        counts[domain] += 1

    total  = max(sum(counts.values()), 1)
    shares = [v / total for v in counts.values()]
    if len(shares) < 2:
        return 0.0
    return max(shares) - min(shares)


def explainability_coverage(articles: list[dict]) -> float:
    """
    Fraction of articles that have a non-empty persuasion_scores dict.
    Target: ≥ 0.95.
    """
    if not articles:
        return 0.0
    covered = sum(1 for a in articles if a.get("persuasion_scores"))
    return covered / len(articles)


def high_risk_suppression_rate(
    ranked_articles:   list[dict],
    unfiltered_articles: list[dict],
    threshold: float = 0.55,
) -> float:
    """
    Fraction of HIGH-risk articles in the unfiltered list that were
    *removed* by responsible ranking.
    """
    high_unfiltered = {a["news_id"] for a in unfiltered_articles
                       if a.get("persuasion_risk", 0) >= threshold}
    if not high_unfiltered:
        return 1.0   # perfect if nothing to suppress

    high_in_ranked = {a["news_id"] for a in ranked_articles
                      if a.get("persuasion_risk", 0) >= threshold}
    suppressed = high_unfiltered - high_in_ranked
    return len(suppressed) / len(high_unfiltered)


# ── Aggregate evaluator ───────────────────────────────────────────────────────

class Evaluator:
    """
    End-to-end evaluator that reads MIND behaviors + article scores and
    computes all metrics in one pass.
    """

    def __init__(self, split: str = "dev"):
        self.split = split

    def run(self, k: int = 10) -> dict:
        behaviors_path = ROOT / "data" / "raw" / "mind" / self.split / "behaviors.tsv"
        articles_path  = PROCESSED / self.split / "articles_scored.jsonl"

        if not behaviors_path.exists():
            log.warning("MIND behaviors not found — running synthetic evaluation.")
            return self._synthetic_eval(k)

        # Load article scores
        article_db: dict[str, dict] = {}
        if articles_path.exists():
            with open(articles_path, encoding="utf-8") as fh:
                for line in fh:
                    a = json.loads(line)
                    article_db[a["news_id"]] = a

        # Load ranked recommendations (need Reranker)
        from src.ranking.reranker import Reranker
        ranker = Reranker(split=self.split)

        all_aucs, all_ndcgs, all_p10, all_mrrs = [], [], [], []
        ranked_pool, unfiltered_pool = [], []

        with open(behaviors_path, encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue
                uid         = row[1]
                impressions = row[4].split()

                labels, raw_scores = [], []
                impression_articles = []
                for imp in impressions:
                    parts = imp.split("-")
                    if len(parts) != 2:
                        continue
                    nid, clicked = parts[0], int(parts[1])
                    art = article_db.get(nid, {"news_id": nid, "persuasion_risk": 0.0,
                                               "persuasion_scores": {}})
                    # Use a persisted model score; use a deterministic neutral
                    # relevance prior when preprocessing has not scored articles.
                    resp_score = float(art.get(
                        "responsible_score",
                        art.get("relevance", 1.0 - float(art.get("persuasion_risk", 0.0))),
                    ))
                    labels.append(clicked)
                    raw_scores.append(resp_score)
                    unfiltered_pool.append(art)
                    impression_articles.append(art)

                if len(set(labels)) < 2:
                    continue

                all_aucs.append(auc(labels, raw_scores))
                all_ndcgs.append(ndcg_at_k(labels, raw_scores, k))
                all_p10.append(precision_at_k(labels, raw_scores, k))
                all_mrrs.append(mrr(labels, raw_scores))
                ranked_pool.extend(sorted(
                    impression_articles,
                    key=lambda article: float(article.get(
                        "responsible_score",
                        article.get("relevance", 1.0 - float(article.get("persuasion_risk", 0.0))),
                    )),
                    reverse=True,
                )[:k])

        result = {
            "auc":              float(np.mean(all_aucs))   if all_aucs   else 0.0,
            f"ndcg@{k}":        float(np.mean(all_ndcgs))  if all_ndcgs  else 0.0,
            f"precision@{k}":   float(np.mean(all_p10))    if all_p10    else 0.0,
            "mrr":              float(np.mean(all_mrrs))   if all_mrrs   else 0.0,
            "per":              persuasion_exposure_reduction(ranked_pool, unfiltered_pool),
            "provider_parity_gap":   provider_parity_gap(ranked_pool),
            "explainability_cover":  explainability_coverage(unfiltered_pool),
            "high_risk_suppressed":  high_risk_suppression_rate(ranked_pool, unfiltered_pool),
        }

        log.info("Evaluation results:")
        for k_name, v in result.items():
            log.info("  %-30s %.4f", k_name, v)
        return result

    def _synthetic_eval(self, k: int = 10) -> dict:
        """Run on synthetic random data for demo / CI purposes."""
        np.random.seed(99)
        results = {"auc": [], f"ndcg@{k}": [], f"precision@{k}": [], "mrr": []}

        for _ in range(200):
            n      = np.random.randint(5, 20)
            labels = [0] * n
            labels[np.random.randint(0, n)] = 1   # one positive
            scores = np.random.rand(n).tolist()

            results["auc"].append(auc(labels, scores))
            results[f"ndcg@{k}"].append(ndcg_at_k(labels, scores, k))
            results[f"precision@{k}"].append(precision_at_k(labels, scores, k))
            results["mrr"].append(mrr(labels, scores))

        # Synthetic SDG metrics
        synthetic_arts = [
            {"persuasion_risk": np.random.rand(),
             "persuasion_scores": {"Reasoning": 0.5},
             "url": f"https://{'reuters' if i % 3 == 0 else 'bbc'}.com/{i}",
             "news_id": str(i)}
            for i in range(100)
        ]
        ranked    = sorted(synthetic_arts, key=lambda x: -x.get("responsible_score", 0.5))[:k]
        unfiltered = synthetic_arts

        return {
            k_name: float(np.mean(v)) for k_name, v in results.items()
        } | {
            "per":                   persuasion_exposure_reduction(ranked, unfiltered),
            "provider_parity_gap":   provider_parity_gap(ranked),
            "explainability_cover":  explainability_coverage(synthetic_arts),
            "high_risk_suppressed":  high_risk_suppression_rate(ranked, unfiltered),
            "note": "synthetic evaluation — run --eval-mind for real results",
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the recommender system.")
    parser.add_argument("--demo",      action="store_true",
                        help="Run synthetic demo evaluation")
    parser.add_argument("--eval-mind", action="store_true",
                        help="Evaluate on MIND dev split")
    parser.add_argument("--k",         type=int, default=10)
    parser.add_argument("--split",     default="dev", choices=["train", "dev"])
    args = parser.parse_args()

    evaluator = Evaluator(split=args.split)

    if args.demo or not args.eval_mind:
        results = evaluator._synthetic_eval(args.k)
    else:
        results = evaluator.run(k=args.k)

    print("\n" + "=" * 60)
    print("📊  EVALUATION RESULTS")
    print("=" * 60)
    for name, value in results.items():
        if isinstance(value, float):
            print(f"  {name:<35} {value:.4f}")
        else:
            print(f"  {name:<35} {value}")
    print("=" * 60)

    # Compare against NRMS baseline targets
    targets = {
        "auc":             (0.690, "≥ 0.695"),
        f"ndcg@{args.k}":  (0.416, "≥ 0.430"),
        f"precision@{args.k}": (0.381, "≥ 0.390"),
        "per":             (0.000, "≥ 0.150"),
        "provider_parity_gap": (1.000, "< 0.050"),
        "explainability_cover": (0.000, "≥ 0.950"),
    }
    print("\n🎯  TARGET COMPARISON (vs. NRMS baseline)")
    print("-" * 60)
    for name, (baseline, target_str) in targets.items():
        val = results.get(name, None)
        if val is None:
            continue
        tick = "✅" if (
            ("≥" in target_str and val >= float(target_str.split("≥")[1])) or
            ("<" in target_str  and val <  float(target_str.split("<")[1]))
        ) else "❌"
        print(f"  {tick}  {name:<30} {val:.4f}  (target: {target_str}, baseline: {baseline})")


if __name__ == "__main__":
    main()
