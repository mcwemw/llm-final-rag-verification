"""Block 5 baselines (M-RAG) — published reference scorers behind the same interface.

Every baseline implements the locked Block 5 interface from `blocks.scorer`:
    score(claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult

Baselines here:
  * PresenceOnlyScorer — citation-presence check (Gao et al., 2023, ALCE); the "floor".
  * (planned) RagasScorer        — LLM-as-judge faithfulness (Es et al., 2024); non-deterministic.
  * (planned) SelfCheckGPTScorer — sampling consistency (Manakul et al., 2023); non-deterministic.
"""
from __future__ import annotations

import time

from blocks.aligner import AlignedClaim
from blocks.scorer import ScorerResult, _claim_id


# ── PresenceOnlyScorer ────────────────────────────────────────────────────────

class PresenceOnlyScorer:
    """Baseline 1 — presence-only citation check (the "floor").

    Mirrors what the course Program 2 / typical RAG tutorials do: it only checks
    that the claim is bound to a citation whose doc_id actually exists among the
    provided documents. It never reads the claim or passage *content*, so it
    cannot detect a claim that cites a real source but says something unsupported.

    score = 1.0  if claim.alignment != "none" and claim.cited_doc_id ∈ {d["doc_id"] for d in cited_docs}
            0.0  otherwise
    Binary, deterministic, zero cost.
    """

    name = "presence-only"

    def score(self, claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult:
        t0 = time.perf_counter()
        known_ids = {d.get("doc_id") for d in cited_docs or []}
        present = (
            claim.alignment != "none"
            and claim.cited_doc_id is not None
            and claim.cited_doc_id in known_ids
        )
        return ScorerResult(
            claim_id=_claim_id(claim),
            method=self.name,
            score=1.0 if present else 0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=0.0,
            deterministic=True,
        )
