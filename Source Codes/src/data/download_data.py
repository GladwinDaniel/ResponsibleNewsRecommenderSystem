"""
download_data.py
----------------
Auto-downloads and caches:
  - SemEval 2023 Task 3 (Persuasion Techniques) via HuggingFace Datasets
  - GoEmotions (emotion labels) via HuggingFace Datasets
  - Clickbait News Detection corpus via Kaggle / direct URLs

MIND dataset must be manually downloaded from https://msnews.github.io/
(requires free Microsoft account + academic licence agreement).
"""

import os
import sys
import zipfile
import tarfile
import shutil
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── Repo root & data dirs ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

DIRS = {
    "semeval2023": RAW / "semeval2023",
    "goEmotions":  RAW / "goEmotions",
    "clickbait":   RAW / "clickbait",
    "mind_train":  RAW / "mind" / "train",
    "mind_dev":    RAW / "mind" / "dev",
}

# ── Remote sources ────────────────────────────────────────────────────────────
# Clickbait corpus — hosted on GitHub (CC0 licence)
CLICKBAIT_URL = (
    "https://github.com/bhargaviparanjape/clickbait/raw/master/dataset/"
    "clickbait_data.gz"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    log.info("Data directories ready.")


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream-download *url* to *dest*, showing a tqdm progress bar."""
    if dest.exists():
        log.info("Already downloaded: %s", dest.name)
        return dest

    log.info("Downloading %s → %s", url, dest)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))

    return dest


def _extract(archive: Path, dest: Path):
    log.info("Extracting %s → %s", archive.name, dest)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.suffix in {".gz", ".tgz", ".tar"}:
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unknown archive format: {archive.suffix}")


# ── Dataset downloaders ───────────────────────────────────────────────────────

def download_semeval(out_dir: Optional[Path] = None):
    """
    Download SemEval-2023 Task 3 (Persuasion Techniques in News) from
    HuggingFace Datasets.  Saves raw JSON/TSV files to *out_dir*.
    """
    out_dir = out_dir or DIRS["semeval2023"]
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".downloaded"
    if marker.exists():
        log.info("SemEval-2023 already present — skipping.")
        return

    try:
        from datasets import load_dataset  # type: ignore
        log.info("Loading SemEval-2023 Task 3 from HuggingFace…")
        ds = load_dataset("SemEvalWorkshop/sem_eval_2023_task_3_subtask_3",
                          trust_remote_code=True)
        for split, data in ds.items():
            path = out_dir / f"{split}.jsonl"
            data.to_json(str(path))
            log.info("  Saved %d rows → %s", len(data), path.name)
        marker.touch()
        log.info("✅  SemEval-2023 download complete.")
    except Exception as exc:
        log.warning("HuggingFace download failed (%s). Creating fallback stub.", exc)
        _create_semeval_stub(out_dir)


def _create_semeval_stub(out_dir: Path):
    """
    If the real dataset is unavailable, write a minimal stub so the pipeline
    can still run in demo mode.
    """
    import json

    TECHNIQUES = [
        "Appeal_to_Authority", "Appeal_to_Fear-Prejudice", "Bandwagon",
        "Black-and-White_Fallacy", "Causal_Oversimplification", "Doubt",
        "Exaggeration-Minimisation", "Flag-Waving", "Glittering_generalities",
        "Loaded_Language", "Name_Calling-Labeling", "Obfuscation",
        "Reasoning", "Red_Herring", "Repetition", "Slogans",
        "Thought-terminating_Clichés", "Transfer", "Whataboutism",
        "Straw_Men", "Appeal_to_Hypocrisy", "Consequential_Oversimplification",
        "Appeal_to_Time"
    ]

    stub_rows = []
    texts = [
        ("SHOCKING: Government HIDING the truth!", ["Appeal_to_Fear-Prejudice", "Loaded_Language"]),
        ("Everyone knows this is the best policy.", ["Bandwagon", "Glittering_generalities"]),
        ("Only two options exist: with us or against us.", ["Black-and-White_Fallacy"]),
        ("The expert said so, therefore it is true.", ["Appeal_to_Authority"]),
        ("This will cause complete societal collapse.", ["Exaggeration-Minimisation", "Appeal_to_Fear-Prejudice"]),
        ("Our nation is the greatest in the world.", ["Flag-Waving"]),
        ("Jobs report shows mixed signals from economy.", []),   # neutral
        ("Scientists publish new climate findings.", []),          # neutral
    ]
    for text, labels in texts:
        row = {"text": text, "labels": labels}
        for t in TECHNIQUES:
            row[t] = 1 if t in labels else 0
        stub_rows.append(row)

    import random, copy
    random.seed(42)
    all_rows = stub_rows * 40          # inflate to 320 rows
    random.shuffle(all_rows)

    splits = {"train": all_rows[:256], "validation": all_rows[256:288], "test": all_rows[288:]}
    for split, rows in splits.items():
        path = out_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        log.info("  Stub %s: %d rows → %s", split, len(rows), path.name)
    (out_dir / ".downloaded").touch()
    log.info("✅  SemEval stub created (use for dev only — get the real dataset for experiments).")


def download_goemotions(out_dir: Optional[Path] = None):
    """Download GoEmotions from HuggingFace Datasets."""
    out_dir = out_dir or DIRS["goEmotions"]
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".downloaded"
    if marker.exists():
        log.info("GoEmotions already present — skipping.")
        return

    try:
        from datasets import load_dataset  # type: ignore
        log.info("Loading GoEmotions from HuggingFace…")
        ds = load_dataset("go_emotions", "simplified")
        for split, data in ds.items():
            path = out_dir / f"{split}.jsonl"
            data.to_json(str(path))
            log.info("  Saved %d rows → %s", len(data), path.name)
        marker.touch()
        log.info("✅  GoEmotions download complete.")
    except Exception as exc:
        log.warning("GoEmotions download failed (%s). Creating stub.", exc)
        _create_goemotions_stub(out_dir)


def _create_goemotions_stub(out_dir: Path):
    import json, random
    emotions = ["admiration", "amusement", "anger", "annoyance", "approval",
                "caring", "confusion", "curiosity", "desire", "disappointment",
                "disapproval", "disgust", "embarrassment", "excitement", "fear",
                "gratitude", "grief", "joy", "love", "nervousness", "optimism",
                "pride", "realization", "relief", "remorse", "sadness",
                "surprise", "neutral"]
    random.seed(0)
    rows = []
    for i in range(200):
        rows.append({"text": f"Stub text {i}", "labels": [random.randint(0, 27)]})
    splits = {"train": rows[:160], "validation": rows[160:180], "test": rows[180:]}
    for split, data in splits.items():
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fh:
            for r in data:
                fh.write(json.dumps(r) + "\n")
    (out_dir / ".downloaded").touch()
    log.info("✅  GoEmotions stub created.")


def download_clickbait(out_dir: Optional[Path] = None):
    """Download clickbait corpus from GitHub."""
    out_dir = out_dir or DIRS["clickbait"]
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".downloaded"
    if marker.exists():
        log.info("Clickbait corpus already present — skipping.")
        return

    try:
        archive = out_dir / "clickbait_data.gz"
        _download_file(CLICKBAIT_URL, archive)
        _extract(archive, out_dir)
        marker.touch()
        log.info("✅  Clickbait download complete.")
    except Exception as exc:
        log.warning("Clickbait download failed (%s). Creating stub.", exc)
        _create_clickbait_stub(out_dir)


def _create_clickbait_stub(out_dir: Path):
    import json
    headlines = [
        ("You won't BELIEVE what happened next!", 1),
        ("Scientists discover potential cancer cure", 0),
        ("This one trick will change your life FOREVER", 1),
        ("Parliament passes new infrastructure bill", 0),
        ("SHOCKING: Celebrity's secret revealed", 1),
        ("Climate report shows rising temperatures", 0),
    ]
    data = [{"headline": h, "clickbait": c} for h, c in headlines * 30]
    path = out_dir / "clickbait_data.jsonl"
    with open(path, "w") as fh:
        for r in data:
            fh.write(json.dumps(r) + "\n")
    (out_dir / ".downloaded").touch()
    log.info("✅  Clickbait stub created.")


def check_mind(train_dir: Optional[Path] = None, dev_dir: Optional[Path] = None):
    """Check whether MIND has been manually placed; print instructions if not."""
    train_dir = train_dir or DIRS["mind_train"]
    dev_dir   = dev_dir   or DIRS["mind_dev"]

    train_ok = (train_dir / "news.tsv").exists() and (train_dir / "behaviors.tsv").exists()
    dev_ok   = (dev_dir   / "news.tsv").exists() and (dev_dir   / "behaviors.tsv").exists()

    if train_ok and dev_ok:
        log.info("✅  MIND dataset found at %s and %s", train_dir, dev_dir)
        return True
    else:
        log.warning(
            "⚠️  MIND not found.\n"
            "    1. Go to https://msnews.github.io/\n"
            "    2. Sign in with a free Microsoft account\n"
            "    3. Download MINDsmall_train.zip + MINDsmall_dev.zip\n"
            "    4. Unzip train into: %s\n"
            "    5. Unzip dev   into: %s\n"
            "    6. Rerun this script.\n"
            "    Tip: run generate_synthetic_data.py for dev/debug without MIND.",
            train_dir, dev_dir,
        )
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for the Responsible News Recommender."
    )
    parser.add_argument(
        "--dataset", choices=["semeval", "goemotions", "clickbait", "mind", "all"],
        default="all", help="Which dataset to download (default: all)"
    )
    args = parser.parse_args()

    _make_dirs()

    if args.dataset in {"semeval", "all"}:
        download_semeval()
    if args.dataset in {"goemotions", "all"}:
        download_goemotions()
    if args.dataset in {"clickbait", "all"}:
        download_clickbait()
    if args.dataset in {"mind", "all"}:
        check_mind()

    log.info("Done. Run preprocess.py next.")


if __name__ == "__main__":
    main()
