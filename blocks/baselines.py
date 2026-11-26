"""Block 5 baselines (M-RAG) — published reference scorers behind the same interface.

Every baseline implements the locked Block 5 interface from `blocks.scorer`:
    score(claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult

Baselines here:
  * PresenceOnlyScorer — citation-presence check (Gao et al., 2023, ALCE); the "floor".
  * RagasScorer        — LLM-as-judge faithfulness (Es et al., 2024) via Groq; non-deterministic.
  * (planned) SelfCheckGPTScorer — sampling consistency (Manakul et al., 2023); non-deterministic.
"""
from __future__ import annotations

import asyncio
import math
import os
import time
import warnings
from typing import Any

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


# ── RagasScorer ───────────────────────────────────────────────────────────────

GROQ_BASE_URL = "https://api.groq.com/openai/v1"   # Groq's OpenAI-compatible endpoint
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"           # same model family as course Program 2


class RagasScorer:
    """Baseline 2 — RAGAS faithfulness, LLM-as-judge (Es et al., 2024).

    How RAGAS scores one claim (2 LLM calls):
      1. statement generation: break the claim into atomic statements;
      2. NLI verdicts: ask the judge LLM whether each statement is supported by the context.
      score = supported statements / all statements   ∈ [0, 1]

    Measurement settings (keep fixed across the study):
      * NO response cache — a cache would replay identical answers and fake Det-var = 0.
      * temperature defaults to 0.0, the setting most favourable to RAGAS; any run-to-run
        variance measured at T=0 is therefore a conservative finding.
      * context = the cited span (same input as the other Block 5 scorers); "all_docs"
        joins every cited_doc instead.
      * Token usage is recorded per call (`last_usage`, `total_usage`) so cost can be
        recomputed later; cost_usd uses the per-million-token prices passed in.

    Interface gap: RAGAS requires the original question (user_input) but the locked
    Block 5 interface passes only (claim, cited_docs). Set it per bundle with
    `set_question()`; if unset we fall back to the claim text and count it in `n_fallback_question`.
    """

    name = "ragas-llm-judge"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        price_in_per_mtok: float | None = None,
        price_out_per_mtok: float | None = None,
        context: str = "cited_span",
        max_retries: int = 6,
        llm: Any = None,
        client: Any = None,
    ) -> None:
        if context not in ("cited_span", "all_docs"):
            raise ValueError("context must be 'cited_span' or 'all_docs'")
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.temperature = temperature
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.context = context
        self.question: str | None = None
        self.n_fallback_question = 0
        self.n_nan = 0
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self._loop = asyncio.new_event_loop()  # one persistent loop: AsyncOpenAI must not hop loops

        if price_in_per_mtok is None or price_out_per_mtok is None:
            warnings.warn(
                "RagasScorer: Groq prices not set -> cost_usd reported as 0.0. Token counts are "
                "still recorded in total_usage; pass price_in_per_mtok / price_out_per_mtok "
                "(USD per 1M tokens, from Groq's pricing page) to report real cost.",
                stacklevel=2,
            )

        from ragas.metrics.collections import Faithfulness

        if llm is None:
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory

            if client is None:
                key = api_key or os.environ.get("GROQ_API_KEY")
                if not key:
                    raise RuntimeError(
                        "GROQ_API_KEY is not set. Run `export GROQ_API_KEY=gsk_...` first "
                        "(never commit the key)."
                    )
                client = AsyncOpenAI(api_key=key, base_url=GROQ_BASE_URL, max_retries=max_retries)
            self._wrap_usage(client)
            kwargs = {} if temperature is None else {"temperature": temperature}
            llm = llm_factory(self.model, provider="openai", client=client, **kwargs)  # no cache
        self._metric = Faithfulness(llm=llm)

    # -- token accounting -----------------------------------------------------
    def _wrap_usage(self, client: Any) -> None:
        """Intercept chat.completions.create to read `usage` from every raw response."""
        completions = client.chat.completions
        orig = completions.create

        async def create(*args: Any, **kwargs: Any) -> Any:
            resp = await orig(*args, **kwargs)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                p = getattr(usage, "prompt_tokens", 0) or 0
                c = getattr(usage, "completion_tokens", 0) or 0
                self.last_usage["prompt_tokens"] += p
                self.last_usage["completion_tokens"] += c
                self.total_usage["prompt_tokens"] += p
                self.total_usage["completion_tokens"] += c
            self.total_usage["calls"] += 1
            return resp

        completions.create = create

    def _cost(self) -> float:
        if self.price_in is None or self.price_out is None:
            return 0.0
        return (self.last_usage["prompt_tokens"] * self.price_in
                + self.last_usage["completion_tokens"] * self.price_out) / 1e6

    # -- interface ------------------------------------------------------------
    def set_question(self, question: str | None) -> None:
        """Set the original question of the bundle being scored (RAGAS user_input)."""
        self.question = question

    @property
    def config(self) -> dict:
        """Settings to log next to results for reproducibility."""
        return {"method": self.name, "model": self.model, "temperature": self.temperature,
                "context": self.context, "cache": None,
                "price_in_per_mtok": self.price_in, "price_out_per_mtok": self.price_out}

    def score(self, claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult:
        t0 = time.perf_counter()
        cid = _claim_id(claim)
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        if self.context == "all_docs":
            contexts = [d.get("text", "") for d in cited_docs or [] if d.get("text")]
        else:
            contexts = [claim.cited_span_text] if claim.cited_span_text else []

        if not cited_docs or claim.alignment == "none" or not contexts or not claim.claim_text:
            return ScorerResult(claim_id=cid, method=self.name, score=0.0,
                                latency_ms=(time.perf_counter() - t0) * 1000,
                                cost_usd=0.0, deterministic=False)

        question = self.question
        if not question:
            question = claim.claim_text
            self.n_fallback_question += 1

        result = self._loop.run_until_complete(
            self._metric.ascore(user_input=question, response=claim.claim_text,
                                retrieved_contexts=contexts)
        )
        value = float(result.value) if result.value is not None else float("nan")
        if math.isnan(value):  # judge produced no statements -> treat as unsupported, but count it
            self.n_nan += 1
            value = 0.0
        return ScorerResult(claim_id=cid, method=self.name,
                            score=max(0.0, min(1.0, value)),
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            cost_usd=self._cost(), deterministic=False)

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.close()
