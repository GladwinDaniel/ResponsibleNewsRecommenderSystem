"""
preprocess.py
-------------
Reads raw MIND + SemEval data, applies:
  1. HTML cleaning & text normalization (ftfy, BeautifulSoup)
  2. Language detection — keeps English only
  3. spaCy NER for entity extraction
  4. Outputs articles.jsonl + users.json to data/processed/{split}/

Usage:
  python src/data/preprocess.py --source mind --split train
  python src/data/preprocess.py --source mind --split dev
  python src/data/preprocess.py --source semeval
"""

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Remove HTML tags, fix encoding glitches, collapse whitespace."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        raw = BeautifulSoup(raw, "lxml").get_text(separator=" ")
    except Exception:
        pass
    try:
        import ftfy  # type: ignore
        raw = ftfy.fix_text(raw)
    except Exception:
        pass
    raw = re.sub(r"[\r\n\t]+", " ", raw)
    raw = re.sub(r"\s{2,}", " ", raw)
    return raw.strip()


def _detect_english(text: str) -> bool:
    try:
        from langdetect import detect  # type: ignore
        return detect(text) == "en"
    except Exception:
        return True   # keep if detection fails


# ── NER ───────────────────────────────────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy  # type: ignore
            _nlp = spacy.load("en_core_web_sm")
            log.info("spaCy model loaded.")
        except Exception as exc:
            log.warning("spaCy not available (%s) — NER disabled.", exc)
            _nlp = False
    return _nlp


def _extract_entities(text: str) -> list[dict]:
    nlp = _get_nlp()
    if not nlp:
        return []
    doc = nlp(text[:10_000])   # cap to avoid slowdowns
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


# ── MIND processing ───────────────────────────────────────────────────────────

def _iter_mind_news(news_tsv: Path) -> Iterator[dict]:
    """Yield raw rows from a MIND news.tsv file."""
    with open(news_tsv, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            yield {
                "news_id":     row[0],
                "category":    row[1].lower(),
                "subcategory": row[2].lower(),
                "title":       row[3],
                "abstract":    row[4] if len(row) > 4 else "",
                "url":         row[5] if len(row) > 5 else "",
            }


def _iter_mind_behaviors(behaviors_tsv: Path) -> Iterator[dict]:
    """Yield raw impression rows from a MIND behaviors.tsv file."""
    with open(behaviors_tsv, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            yield {
                "impression_id": row[0],
                "user_id":       row[1],
                "time":          row[2],
                "history":       row[3].split() if row[3] else [],
                "impressions":   row[4].split() if len(row) > 4 else [],
            }


def process_mind(split: str = "train"):
    mind_dir  = RAW / "mind" / split
    out_dir   = PROCESSED / split
    out_dir.mkdir(parents=True, exist_ok=True)

    news_tsv      = mind_dir / "news.tsv"
    behaviors_tsv = mind_dir / "behaviors.tsv"

    if not news_tsv.exists():
        log.error("MIND not found at %s. Run generate_synthetic_data.py first.", news_tsv)
        return

    # ── Process articles ──────────────────────────────────────────────────────
    log.info("Processing articles from %s …", news_tsv)
    articles_out = out_dir / "articles.jsonl"
    count = 0
    with open(articles_out, "w", encoding="utf-8") as fh:
        for raw in _iter_mind_news(news_tsv):
            title    = _clean_text(raw["title"])
            abstract = _clean_text(raw["abstract"])
            full_text = f"{title}. {abstract}"

            if not _detect_english(full_text):
                continue

            entities = _extract_entities(full_text)

            article = {
                "news_id":    raw["news_id"],
                "category":   raw["category"],
                "subcategory":raw["subcategory"],
                "title":      title,
                "abstract":   abstract,
                "url":        raw.get("url", ""),
                "entities":   entities,
                # Persuasion scores filled by persuasion_detector.py
                "persuasion_scores": {},
                "persuasion_risk":   0.0,
            }
            fh.write(json.dumps(article) + "\n")
            count += 1

    log.info("  ✅  %d articles written → %s", count, articles_out)

    # ── Process users ─────────────────────────────────────────────────────────
    if not behaviors_tsv.exists():
        log.warning("behaviors.tsv not found — skipping user extraction.")
        return

    log.info("Processing behaviors from %s …", behaviors_tsv)
    users: dict[str, dict] = defaultdict(lambda: {"history": [], "impressions": []})

    for row in _iter_mind_behaviors(behaviors_tsv):
        uid = row["user_id"]
        users[uid]["history"].extend(row["history"])
        clicked = [imp.split("-")[0] for imp in row["impressions"] if imp.endswith("-1")]
        users[uid]["impressions"].extend(clicked)

    # Deduplicate and trim history
    for uid, data in users.items():
        data["history"]     = list(dict.fromkeys(data["history"]))[-50:]
        data["impressions"] = list(dict.fromkeys(data["impressions"]))

    users_out = out_dir / "users.json"
    with open(users_out, "w", encoding="utf-8") as fh:
        json.dump(users, fh)

    log.info("  ✅  %d users written → %s", len(users), users_out)


# ── SemEval processing ────────────────────────────────────────────────────────

def process_semeval():
    semeval_dir = RAW / "semeval2023"
    out_dir     = PROCESSED / "semeval"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    for split in ("train", "validation", "test"):
        src = semeval_dir / f"{split}.jsonl"
        if not src.exists():
            log.warning("SemEval %s split not found — skipping.", split)
            continue

        out = out_dir / f"{split}.jsonl"
        processed_rows = []
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                text   = _clean_text(row.get("text", ""))
                labels = [t for t in TECHNIQUES if row.get(t, 0) == 1]
                processed_rows.append({"text": text, "labels": labels,
                                       "binary": [row.get(t, 0) for t in TECHNIQUES]})

        with open(out, "w", encoding="utf-8") as fh:
            for r in processed_rows:
                fh.write(json.dumps(r) + "\n")
        log.info("  ✅  SemEval %s: %d rows → %s", split, len(processed_rows), out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw datasets.")
    parser.add_argument("--source", choices=["mind", "semeval", "all"], default="all")
    parser.add_argument("--split",  choices=["train", "dev"], default="train",
                        help="MIND split to process (only used when --source=mind)")
    args = parser.parse_args()

    if args.source in {"mind", "all"}:
        process_mind(args.split)
        if args.source == "all":
            process_mind("dev")
    if args.source in {"semeval", "all"}:
        process_semeval()

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
