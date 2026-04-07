"""
detector.py — ML inference engine

Loads two RoBERTa-based AI-text-detection models and ensembles their scores.

Primary:   openai-community/roberta-large-openai-detector (RoBERTa Large, 355M params)
Secondary: openai-community/roberta-base-openai-detector  (RoBERTa Base,  125M params)

Both models output {Real, Fake} labels and have a 512-token input limit,
so longer code is split into overlapping chunks and scores are aggregated
via confidence-weighted average.
"""

import time
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

log = logging.getLogger("detector")

PRIMARY_MODEL   = "openai-community/roberta-large-openai-detector"
SECONDARY_MODEL = "openai-community/roberta-base-openai-detector"

MAX_TOKENS    = 512
CHUNK_STRIDE  = 256
ENSEMBLE_W    = {"primary": 0.65, "secondary": 0.35}

_models: dict = {}


def _load_model(name: str):
    """Load a single HuggingFace sequence-classification model + tokenizer."""
    log.info("Loading model %s …", name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


def load_models():
    """Download / cache both models. Called once at startup."""
    if _models:
        return
    for key, name in [("primary", PRIMARY_MODEL), ("secondary", SECONDARY_MODEL)]:
        tokenizer, model = _load_model(name)
        _models[key] = {"tokenizer": tokenizer, "model": model, "name": name}
    log.info("All models loaded.")


def models_ready() -> bool:
    return "primary" in _models and "secondary" in _models


def _chunk_code(code: str, tokenizer, max_tokens: int = MAX_TOKENS, stride: int = CHUNK_STRIDE):
    """
    Tokenise the full code and yield overlapping windows of token IDs,
    each up to *max_tokens* long, advancing by *stride* tokens.
    """
    token_ids = tokenizer.encode(code, add_special_tokens=False)

    if len(token_ids) <= max_tokens - 2:
        yield token_ids
        return

    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens - 2, len(token_ids))
        yield token_ids[start:end]
        if end >= len(token_ids):
            break
        start += stride


def _infer_single_model(key: str, code: str) -> dict:
    """
    Run one model over all chunks of *code* and return an aggregated result.

    Returns dict with: score (0-100, high = AI), confidence (0-100),
    label (str), raw per-chunk scores.
    """
    entry     = _models[key]
    tokenizer = entry["tokenizer"]
    model     = entry["model"]
    device    = next(model.parameters()).device

    chunks       = list(_chunk_code(code, tokenizer))
    chunk_scores = []

    for chunk_ids in chunks:
        input_ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
        input_ids = torch.tensor([input_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[0]

        id2label = model.config.id2label or {}
        labels   = [id2label.get(i, str(i)) for i in range(probs.shape[0])]

        ai_idx = _find_ai_index(labels)
        ai_prob = probs[ai_idx].item()
        conf    = abs(ai_prob - 0.5) * 200

        chunk_scores.append({"ai_prob": ai_prob, "confidence": conf})

    if not chunk_scores:
        return {"score": 50, "confidence": 0, "label": "unknown"}

    total_conf = sum(c["confidence"] for c in chunk_scores) or 1.0
    weighted_prob = sum(
        c["ai_prob"] * (c["confidence"] / total_conf) for c in chunk_scores
    )

    score = min(100, max(0, round(weighted_prob * 100)))
    confidence = min(99, round(sum(c["confidence"] for c in chunk_scores) / len(chunk_scores)))
    label = "ai" if score >= 50 else "human"

    return {
        "score": score,
        "confidence": confidence,
        "label": label,
        "model": entry["name"],
        "chunks": len(chunk_scores),
    }


def _find_ai_index(labels: list[str]) -> int:
    """
    Determine which output index corresponds to "AI / Fake / Synthetic".
    Different models use different label names.
    """
    for i, lab in enumerate(labels):
        low = lab.lower()
        if low in ("fake", "ai", "ai-generated", "synthetic", "machine", "1"):
            return i
    # Fallback: assume last index is the positive (AI) class
    return len(labels) - 1


def analyze(code: str, language: str = "Python") -> dict:
    """
    Run both models, ensemble scores, return unified result.
    """
    if not models_ready():
        raise RuntimeError("Models not loaded")

    t0 = time.perf_counter()

    primary   = _infer_single_model("primary", code)
    secondary = _infer_single_model("secondary", code)

    wp = ENSEMBLE_W["primary"]
    ws = ENSEMBLE_W["secondary"]
    ensemble_score = round(primary["score"] * wp + secondary["score"] * ws)
    ensemble_score = min(100, max(0, ensemble_score))

    ensemble_conf = round(primary["confidence"] * wp + secondary["confidence"] * ws)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    return {
        "score": ensemble_score,
        "confidence": ensemble_conf,
        "models": {
            "primary": primary,
            "secondary": secondary,
        },
        "chunks_analyzed": max(primary["chunks"], secondary["chunks"]),
        "processing_time_ms": elapsed_ms,
    }
