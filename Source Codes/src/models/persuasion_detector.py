"""
persuasion_detector.py
----------------------
Fine-tunes RoBERTa-base for 23-class multi-label persuasion technique
detection (SemEval-2023 Task 3, Subtask 3).

Outputs
-------
  checkpoints/persuasion_roberta_best.pt  — best model weights
  data/processed/train/articles_scored.jsonl — MIND articles + scores
  data/processed/dev/articles_scored.jsonl

Usage
-----
  # Train (RTX 5060, ~4-6 h on real SemEval):
  python src/models/persuasion_detector.py --train

  # Infer on all MIND articles (requires trained checkpoint):
  python src/models/persuasion_detector.py --infer

  # Demo (no training required):
  python src/models/persuasion_detector.py --demo "SHOCKING: hidden truth revealed!"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
PROCESSED   = ROOT / "data" / "processed"
CHECKPOINTS.mkdir(exist_ok=True)

# ── 23 SemEval persuasion technique labels ────────────────────────────────────
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
NUM_LABELS = len(TECHNIQUES)   # 23

# Per-technique "danger weight" — used to compute composite persuasion risk
DANGER_WEIGHTS = np.array([
    0.6,  # Appeal_to_Authority
    0.9,  # Appeal_to_Fear-Prejudice
    0.7,  # Bandwagon
    0.8,  # Black-and-White_Fallacy
    0.7,  # Causal_Oversimplification
    0.8,  # Doubt
    0.6,  # Exaggeration-Minimisation
    0.7,  # Flag-Waving
    0.5,  # Glittering_generalities
    0.9,  # Loaded_Language
    0.85, # Name_Calling-Labeling
    0.6,  # Obfuscation
    0.3,  # Reasoning  (reasoning is GOOD — low risk)
    0.5,  # Red_Herring
    0.4,  # Repetition
    0.5,  # Slogans
    0.7,  # Thought-terminating_Clichés
    0.6,  # Transfer
    0.7,  # Whataboutism
    0.8,  # Straw_Men
    0.75, # Appeal_to_Hypocrisy
    0.6,  # Consequential_Oversimplification
    0.3,  # Appeal_to_Time
], dtype=np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SemEvalDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_len: int = 256):
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row    = self.rows[idx]
        text   = row["text"]
        labels = row.get("binary", [0] * NUM_LABELS)
        enc    = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(labels, dtype=torch.float32),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class PersuasionDetector(nn.Module):
    """RoBERTa-base with a multi-label classification head (23 outputs)."""

    def __init__(self, model_name: str = "roberta-base", num_labels: int = NUM_LABELS,
                 dropout: float = 0.1):
        super().__init__()
        try:
            from transformers import RobertaModel  # type: ignore
            self.encoder  = RobertaModel.from_pretrained(model_name)
            self.hidden   = self.encoder.config.hidden_size   # 768
        except Exception:
            log.warning("RoBERTa not available — using lightweight fallback encoder.")
            self.encoder = None
            self.hidden  = 256
            self._fallback_embed = nn.Embedding(30_000, self.hidden)
            self._fallback_pool  = nn.Linear(self.hidden, self.hidden)

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        if self.encoder is not None:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0, :]          # [CLS] token
        else:
            emb    = self._fallback_embed(input_ids)         # (B, L, H)
            pooled = emb.mean(dim=1)                          # mean pooling
            pooled = torch.relu(self._fallback_pool(pooled))

        pooled = self.dropout(pooled)
        return self.classifier(pooled)                        # (B, 23)

    def predict_proba(self, input_ids, attention_mask) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            return torch.sigmoid(logits).cpu().numpy()


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    semeval_train: Optional[Path] = None,
    semeval_val:   Optional[Path] = None,
    epochs:        int = 5,
    batch_size:    int = 16,
    lr:            float = 2e-5,
    max_len:       int = 256,
    model_name:    str = "roberta-base",
):
    semeval_train = semeval_train or (PROCESSED / "semeval" / "train.jsonl")
    semeval_val   = semeval_val   or (PROCESSED / "semeval" / "validation.jsonl")

    if not semeval_train.exists():
        log.error("SemEval train data not found at %s. Run download_data.py first.", semeval_train)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on: %s", device)

    try:
        from transformers import RobertaTokenizerFast  # type: ignore
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    except Exception:
        log.warning("RoBERTa tokenizer unavailable — using fallback.")
        tokenizer = _FallbackTokenizer(max_len)

    train_ds  = SemEvalDataset(semeval_train, tokenizer, max_len)
    val_ds    = SemEvalDataset(semeval_val,   tokenizer, max_len) if semeval_val.exists() else None

    train_dl  = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl    = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None

    model     = PersuasionDetector(model_name=model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    ckpt_path     = CHECKPOINTS / "persuasion_roberta_best.pt"

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [train]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_dl)

        # Validate
        if val_dl:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    input_ids      = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels         = batch["labels"].to(device)
                    logits         = model(input_ids, attention_mask)
                    val_loss      += criterion(logits, labels).item()
            avg_val = val_loss / len(val_dl)
            log.info("Epoch %d | train_loss=%.4f | val_loss=%.4f", epoch, avg_train, avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), ckpt_path)
                log.info("  ✅  Best model saved → %s", ckpt_path)
        else:
            log.info("Epoch %d | train_loss=%.4f", epoch, avg_train)
            torch.save(model.state_dict(), ckpt_path)

    log.info("Training complete. Best checkpoint: %s", ckpt_path)


# ── Inference on MIND articles ────────────────────────────────────────────────

def infer(split: str = "train", batch_size: int = 64, max_len: int = 128):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CHECKPOINTS / "persuasion_roberta_best.pt"

    try:
        from transformers import RobertaTokenizerFast  # type: ignore
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    except Exception:
        tokenizer = _FallbackTokenizer(max_len)

    model = PersuasionDetector()
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        log.info("Loaded checkpoint from %s", ckpt_path)
    else:
        log.warning("No checkpoint found — using random weights (for demo only).")
    model = model.to(device)
    model.eval()

    articles_in  = PROCESSED / split / "articles.jsonl"
    articles_out = PROCESSED / split / "articles_scored.jsonl"

    if not articles_in.exists():
        log.error("articles.jsonl not found. Run preprocess.py first.")
        return

    log.info("Scoring articles from %s …", articles_in)
    articles = []
    with open(articles_in, encoding="utf-8") as fh:
        for line in fh:
            articles.append(json.loads(line))

    with open(articles_out, "w", encoding="utf-8") as fh:
        for i in range(0, len(articles), batch_size):
            batch    = articles[i : i + batch_size]
            texts    = [f"{a['title']}. {a['abstract']}" for a in batch]
            enc      = tokenizer(texts, max_length=max_len, padding="max_length",
                                 truncation=True, return_tensors="pt")
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                probs = torch.sigmoid(model(input_ids, attention_mask)).cpu().numpy()

            for article, prob_vec in zip(batch, probs):
                scores = {t: float(p) for t, p in zip(TECHNIQUES, prob_vec)}
                risk   = float(np.dot(prob_vec, DANGER_WEIGHTS) / DANGER_WEIGHTS.sum())
                article["persuasion_scores"] = scores
                article["persuasion_risk"]   = risk
                fh.write(json.dumps(article) + "\n")

    log.info("✅  Scored %d articles → %s", len(articles), articles_out)


# ── Demo (single text) ────────────────────────────────────────────────────────

_demo_model = None
_demo_tokenizer = None

def get_demo_detector():
    """Lazy singleton getter for the persuasion detector model."""
    global _demo_model, _demo_tokenizer
    if _demo_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import RobertaTokenizerFast  # type: ignore
            _demo_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        except Exception:
            _demo_tokenizer = _FallbackTokenizer(128)

        ckpt_path = CHECKPOINTS / "persuasion_roberta_best.pt"
        _demo_model = PersuasionDetector()
        if ckpt_path.exists():
            _demo_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        _demo_model = _demo_model.to(device).eval()
    return _demo_model, _demo_tokenizer


def predict_text(text: str) -> tuple[np.ndarray, float]:
    """Score any text and return (23-dim probability vector, overall risk score)."""
    model, tokenizer = get_demo_detector()
    device = next(model.parameters()).device
    enc = tokenizer(text, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        probs = torch.sigmoid(
            model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        ).cpu().numpy()[0]
    risk = float(np.dot(probs, DANGER_WEIGHTS) / DANGER_WEIGHTS.sum())
    return probs, risk


def highlight_spans(text: str) -> list[dict]:
    """
    Split text into sentence-level spans and evaluate fine-grained persuasion techniques
    for deep research attribution and frontend highlighting.
    """
    import re
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text.strip()] if text.strip() else []

    spans = []
    for s in raw_sentences:
        probs, r = predict_text(s)
        detected = [
            {"technique": t, "score": round(float(p), 3)}
            for t, p in zip(TECHNIQUES, probs) if p > 0.3
        ]
        detected.sort(key=lambda x: -x["score"])
        sev = "low" if r < 0.25 else ("medium" if r < 0.55 else "high")
        spans.append({
            "text": s,
            "risk": round(float(r), 3),
            "severity": sev,
            "detected_techniques": detected[:3],
            "is_flagged": r >= 0.35,
        })
    return spans


def demo(text: str):
    probs, risk = predict_text(text)
    print(f"\n📰  Text: {text}\n")
    print(f"⚠️  Persuasion Risk Score: {risk:.3f}\n")
    print("Detected Techniques (prob > 0.3):")
    detected = [(t, float(p)) for t, p in zip(TECHNIQUES, probs) if p > 0.3]
    if detected:
        for t, p in sorted(detected, key=lambda x: -x[1]):
            bar = "█" * int(p * 20)
            print(f"  {t:<42} {p:.3f}  {bar}")
    else:
        print("  None detected above threshold (0.3).")
    return probs, risk


# ── Fallback tokenizer (no HuggingFace) ──────────────────────────────────────

class _FallbackTokenizer:
    """Minimal character-level tokenizer for offline testing."""
    def __init__(self, max_len: int = 128):
        self.max_len = max_len

    def __call__(self, text_or_texts, max_length=None, padding=None,
                 truncation=None, return_tensors=None):
        max_length = max_length or self.max_len
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = text_or_texts

        ids_list, mask_list = [], []
        for t in texts:
            ids  = [ord(c) % 30_000 for c in t[:max_length]]
            pad  = max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad
            ids  = ids + [0] * pad
            ids_list.append(ids)
            mask_list.append(mask)

        return {
            "input_ids":      torch.tensor(ids_list,  dtype=torch.long),
            "attention_mask": torch.tensor(mask_list, dtype=torch.long),
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Persuasion technique detector (RoBERTa).")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train",  action="store_true", help="Fine-tune on SemEval-2023")
    group.add_argument("--infer",  action="store_true", help="Score all MIND articles")
    group.add_argument("--demo",   type=str, metavar="TEXT", help="Score a single text")
    parser.add_argument("--split",       default="train", choices=["train", "dev"])
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.infer:
        infer(split=args.split, batch_size=args.batch_size * 4)
    elif args.demo:
        demo(args.demo)


if __name__ == "__main__":
    main()
