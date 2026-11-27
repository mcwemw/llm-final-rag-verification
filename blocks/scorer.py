"""Block 5 — Grounding Scorer: unified interface + three implementations.

All scorers expose:
    score(claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult

cited_docs: list of {doc_id, text} dicts (citations or retrieved passages).
When claim.alignment == "none" the scorer returns score=0.0 immediately;
Block 6 maps alignment==none → UNSOURCED regardless of score.

Rule set locked 2026-11-24.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from blocks.aligner import AlignedClaim


@dataclass(frozen=True)
class ScorerResult:
    claim_id: str       # surrogate: f"c{claim.char_start}"
    method: str         # "bge-m3-cosine" | "nli-true" | "hw1-selfattn"
    score: float        # [0.0, 1.0]; higher = better grounded
    latency_ms: float   # wall-clock ms
    cost_usd: float     # 0.0 for all local scorers
    deterministic: bool = True


@runtime_checkable
class Scorer(Protocol):
    name: str
    def score(self, claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult: ...


def _claim_id(claim: AlignedClaim) -> str:
    return f"c{claim.char_start}"


def _cited_text(claim: AlignedClaim) -> str:
    return claim.cited_span_text or ""


# ── EmbeddingScorer ────────────────────────────────────────────────────────────

class EmbeddingScorer:
    """bge-m3 cosine similarity. Local CPU, deterministic (fixed weights, no sampling)."""

    name = "bge-m3-cosine"

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

    def score(self, claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult:
        t0 = time.perf_counter()
        cid = _claim_id(claim)
        cited = _cited_text(claim)

        if not cited_docs or claim.alignment == "none" or not cited:
            return ScorerResult(
                claim_id=cid, method=self.name,
                score=0.0, latency_ms=(time.perf_counter()-t0)*1000,
                cost_usd=0.0, deterministic=True,
            )

        embs = self._model.encode(
            [claim.claim_text, cited],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        s = float(embs[0] @ embs[1])
        s = max(0.0, min(1.0, s))
        return ScorerResult(
            claim_id=cid, method=self.name,
            score=s, latency_ms=(time.perf_counter()-t0)*1000,
            cost_usd=0.0, deterministic=True,
        )


# ── NLIScorer ─────────────────────────────────────────────────────────────────

def _softmax(x: "np.ndarray") -> "np.ndarray":
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()


def _label_index(id2label: dict, name: str) -> int:
    """Index of `name` in a model's id2label (case-insensitive). Fail loud if it is absent."""
    hits = [int(i) for i, lab in id2label.items() if str(lab).lower() == name]
    if len(hits) != 1:
        raise ValueError(f"NLI model has no unique {name!r} label: {id2label}")
    return hits[0]


class NLIScorer:
    """Cross-encoder NLI entailment probability. Local, deterministic."""

    name = "nli-true"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, device="cpu")
        # Look the entailment index up by name: label order differs between NLI models
        # (cross-encoder/nli-deberta-v3-* is 0=contradiction, 1=entailment, 2=neutral).
        self._entail_idx = _label_index(self._model.config.id2label, "entailment")

    def score(self, claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult:
        t0 = time.perf_counter()
        cid = _claim_id(claim)
        cited = _cited_text(claim)

        if not cited_docs or claim.alignment == "none" or not cited:
            return ScorerResult(
                claim_id=cid, method=self.name,
                score=0.0, latency_ms=(time.perf_counter()-t0)*1000,
                cost_usd=0.0, deterministic=True,
            )

        # premise = cited span, hypothesis = claim
        logits = self._model.predict([(cited, claim.claim_text)])
        probs = _softmax(logits[0])
        s = float(probs[self._entail_idx])  # entailment probability
        s = max(0.0, min(1.0, s))
        return ScorerResult(
            claim_id=cid, method=self.name,
            score=s, latency_ms=(time.perf_counter()-t0)*1000,
            cost_usd=0.0, deterministic=True,
        )


# ── SelfAttentionScorer ───────────────────────────────────────────────────────

class SelfAttentionScorer:
    """HW1 from-scratch self-attention cosine regressor.

    Implements the same character n-gram → self-attention pooling pipeline
    as HW1 P2.  No external ML framework required; numpy only.
    Fixed random seed → fully deterministic.
    """

    name = "hw1-selfattn"

    _DIM = 64
    _NGRAMS = (2, 3)

    def __init__(self) -> None:
        import numpy as np
        self._np = np
        # Fixed projection weights (seed=42, never retrained here — mirrors HW1 static embed)
        rng = np.random.default_rng(42)
        self._W_q = rng.standard_normal((self._DIM, self._DIM)).astype(np.float32)
        self._W_k = rng.standard_normal((self._DIM, self._DIM)).astype(np.float32)
        self._W_v = rng.standard_normal((self._DIM, self._DIM)).astype(np.float32)

    def _ngram_embed(self, text: str) -> "np.ndarray":
        np = self._np
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return np.zeros(self._DIM, dtype=np.float32)
        vecs = []
        for w in words:
            v = np.zeros(self._DIM, dtype=np.float32)
            for n in self._NGRAMS:
                for i in range(len(w) - n + 1):
                    h = hash(w[i:i+n]) % self._DIM
                    v[h] += 1.0
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.stack(vecs)  # (T, DIM)

    def _self_attention_pool(self, X: "np.ndarray") -> "np.ndarray":
        """Single-head self-attention → mean pooling over attended values."""
        np = self._np
        if X.ndim == 1:
            X = X[None, :]
        Q = X @ self._W_q.T          # (T, DIM)
        K = X @ self._W_k.T
        V = X @ self._W_v.T
        scale = self._DIM ** 0.5
        A = np.exp((Q @ K.T) / scale)
        A = A / (A.sum(axis=-1, keepdims=True) + 1e-9)  # softmax
        out = A @ V                  # (T, DIM)
        pooled = out.mean(axis=0)    # (DIM,)
        norm = np.linalg.norm(pooled)
        return pooled / norm if norm > 0 else pooled

    def score(self, claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult:
        t0 = time.perf_counter()
        np = self._np
        cid = _claim_id(claim)
        cited = _cited_text(claim)

        if not cited_docs or claim.alignment == "none" or not cited:
            return ScorerResult(
                claim_id=cid, method=self.name,
                score=0.0, latency_ms=(time.perf_counter()-t0)*1000,
                cost_usd=0.0, deterministic=True,
            )

        X_claim = self._ngram_embed(claim.claim_text)
        X_cited = self._ngram_embed(cited)

        v_claim = self._self_attention_pool(X_claim)
        v_cited = self._self_attention_pool(X_cited)

        s = float(np.dot(v_claim, v_cited))
        s = max(0.0, min(1.0, s))
        return ScorerResult(
            claim_id=cid, method=self.name,
            score=s, latency_ms=(time.perf_counter()-t0)*1000,
            cost_usd=0.0, deterministic=True,
        )
