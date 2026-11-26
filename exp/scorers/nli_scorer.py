"""NLI-based faithfulness scorer (cross-encoder).

Uses a cross-encoder NLI model to judge entailment:
  premise = context (retrieved passages)
  hypothesis = response/answer

Returns entailment probability [0, 1]. Fully deterministic (no sampling).
determinism_var = 0 (same input → same logits).

Model: cross-encoder/nli-MiniLM2-L6-H768 (~90MB, CPU-friendly, BERT fast tokenizer).
(Was cross-encoder/nli-deberta-v3-small, but its tokenizer.json is incompatible with the
pinned `tokenizers`/transformers 4.39.3 on Intel Mac — see _pipeline() note.)

Cite: Hönovich et al., 2022. "True: Re-evaluating Factual Consistency Evaluation."
"""
from __future__ import annotations
import numpy as np

_PIPE = None


def _pipeline():
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline
        # NOTE: deberta-v3-small's tokenizer.json needs a newer `tokenizers` than the
        # pinned env (transformers 4.39.3 / torch 2.2.2 on Intel Mac); loading it raises
        # "PyPreTokenizerTypeWrapper" and the slow tokenizer needs SentencePiece (no wheel,
        # cmake build). We use the docstring's recommended fallback cross-encoder
        # (BERT fast tokenizer, no SentencePiece) — same NLI-entailment technique.
        _PIPE = pipeline(
            "text-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
            device=-1,         # CPU — no MPS, no CUDA
            top_k=None,        # return all labels
        )
    return _PIPE


def _entail_prob(result: list[dict]) -> float:
    """Extract ENTAILMENT probability from pipeline output."""
    for item in result:
        if item["label"].upper() in ("ENTAILMENT", "LABEL_2"):
            return item["score"]
    # fallback: use 1 - contradiction
    for item in result:
        if item["label"].upper() in ("CONTRADICTION", "LABEL_0"):
            return 1.0 - item["score"]
    return 0.5


def score_batch(records: list[dict], batch_size: int = 8) -> list[float]:
    """
    Returns entailment score per record in [0,1].
    premise = source_text (context), hypothesis = response.
    """
    pipe = _pipeline()
    pairs = [
        {"text": rec.get("source_text", str(rec.get("passages", "")))[:512],
         "text_pair": rec.get("response", rec.get("response_text", ""))[:256]}
        for rec in records
    ]
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        results = pipe(batch, truncation=True, max_length=512)
        for res in results:
            if isinstance(res, list):
                scores.append(_entail_prob(res))
            else:
                scores.append(_entail_prob([res]))
    return scores


def threshold_scores(scores: list[float], t: float = 0.5) -> list[int]:
    """Binary verdict: entailment < t → flagged (hallucinated=1)."""
    return [1 if s < t else 0 for s in scores]
