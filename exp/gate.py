"""Deterministic embedding-cosine grounding scorer + 4-way faithfulness gate.

PASS / WARN / FAIL / UNSOURCED. Fully deterministic: same input -> same cosine
-> same label (variance = 0 across runs). bge-m3 forced to CPU (Intel Mac, no MPS).
"""
from __future__ import annotations
import functools
import numpy as np

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("BAAI/bge-m3", device="cpu")
    return _MODEL


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalized embeddings (bge-m3, CPU). Deterministic."""
    m = _model()
    return m.encode(
        texts, batch_size=16, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )


def cosine_pairs(claims: list[str], sources: list[str]) -> np.ndarray:
    """Per-pair cosine between claim[i] and its cited source[i]."""
    assert len(claims) == len(sources)
    ce = embed(claims)
    se = embed(sources)
    return np.sum(ce * se, axis=1)  # already normalized


# default band: calibrated later on a held-out split. Seed from prior bge-m3 work.
HI_DEFAULT = 0.60
LO_DEFAULT = 0.45


def classify(cosine: float, citation_present: bool,
             hi: float = HI_DEFAULT, lo: float = LO_DEFAULT) -> str:
    if not citation_present:
        return "UNSOURCED"
    if cosine >= hi:
        return "PASS"
    if cosine >= lo:
        return "WARN"
    return "FAIL"


def is_flagged(label: str) -> bool:
    """Detector fires (predicts 'not grounded') for everything but PASS."""
    return label != "PASS"
