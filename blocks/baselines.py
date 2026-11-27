"""Block 5 baselines (M-RAG) — published reference scorers behind the same interface.

Every baseline implements the locked Block 5 interface from `blocks.scorer`:
    score(claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult

Baselines here:
  * PresenceOnlyScorer — citation-presence check (Gao et al., 2023, ALCE); the "floor".
  * RagasScorer        — LLM-as-judge faithfulness (Es et al., 2024) via Groq; non-deterministic.
  * SelfCheckGPTScorer — sampling consistency, NLI variant (Manakul et al., 2023); non-deterministic.
"""
from __future__ import annotations

import asyncio
import json
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




# ── LLM provider settings (shared by RAGAS and SelfCheckGPT) ──────────────────

# OpenAI-compatible providers. Never mix providers or models inside one set of reruns: a
# different serving stack / model is a different experimental condition.
PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "model": "openai/gpt-oss-20b"},
    # gpt-oss-20b / -120b were removed from Fireworks serverless (2026-08-27 / 2026-11-07);
    # GLM-5.3-Flash (open weights, zai-org/GLM-5.3-Flash) chosen in the pilot comparison.
    "fireworks": {"base_url": "https://api.fireworks.ai/inference/v1", "key_env": "FIREWORKS_API_KEY",
                  "model": "accounts/fireworks/models/glm-5p3-flash"},
    # NVIDIA NIM hosted API (judge pilot 2026-11-27). Free-tier terms / RPM / retention: UNKNOWN.
    "nim": {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY",
            "model": "mistralai/mistral-large-4-1t-a52b"},
}
GROQ_BASE_URL = PROVIDERS["groq"]["base_url"]
DEFAULT_GROQ_MODEL = PROVIDERS["groq"]["model"]


def _provider(name: str) -> dict[str, str]:
    if name not in PROVIDERS:
        raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
    return PROVIDERS[name]


def _api_key(provider: str, api_key: str | None) -> str:
    env = _provider(provider)["key_env"]
    key = api_key or os.environ.get(env)
    if not key:
        raise RuntimeError(f"{env} is not set. Put it in the environment or .env (never commit the key).")
    return key


class _Usage:
    """Token / call / HTTP-request counters. retries = http_requests - calls."""

    def __init__(self) -> None:
        self.total = self._zero()
        self.last = self._zero()

    @staticmethod
    def _zero() -> dict:
        return {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "calls": 0,
                "http_requests": 0}

    def reset_last(self) -> None:
        self.last = self._zero()

    def add(self, key: str, n: int) -> None:
        self.last[key] += n
        self.total[key] += n

    def add_response(self, resp: Any) -> None:
        self.add("calls", 1)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.add("prompt_tokens", getattr(usage, "prompt_tokens", 0) or 0)
            self.add("completion_tokens", getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            self.add("reasoning_tokens", getattr(details, "reasoning_tokens", 0) or 0)

    def cost(self, d: dict, price_in: float | None, price_out: float | None) -> float:
        if price_in is None or price_out is None:
            return 0.0
        return (d["prompt_tokens"] * price_in + d["completion_tokens"] * price_out) / 1e6

    @property
    def last_retries(self) -> int:
        return max(0, self.last["http_requests"] - self.last["calls"])


def _warn_prices(cls: str, price_in, price_out) -> None:
    if price_in is None or price_out is None:
        warnings.warn(f"{cls}: prices not set -> cost_usd reported as 0.0. Token counts are still "
                      "recorded; pass price_in_per_mtok / price_out_per_mtok (USD per 1M tokens).",
                      stacklevel=3)


def _call_record(resp: Any, request_kwargs: dict, latency_ms: float) -> dict:
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
    choice = (getattr(resp, "choices", None) or [None])[0]
    msg = getattr(choice, "message", None)
    content = getattr(msg, "content", None)
    try:
        json.loads(content or "")
        json_valid = True
    except ValueError:
        json_valid = False
    rf = request_kwargs.get("response_format")
    return {"model_returned": getattr(resp, "model", None),
            "response_format": rf.get("type") if isinstance(rf, dict) else rf,
            "finish_reason": getattr(choice, "finish_reason", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "latency_ms": round(latency_ms, 1), "json_valid": json_valid, "content": content}


# ── RagasScorer ───────────────────────────────────────────────────────────────

class RagasScorer:
    """Baseline 2 — RAGAS faithfulness, LLM-as-judge (Es et al., 2024).

    How RAGAS scores one claim (2 LLM calls):
      1. statement generation: break the claim into atomic statements;
      2. NLI verdicts: ask the judge LLM whether each statement is supported by the context.
      score = supported statements / all statements   ∈ [0, 1]

    Measurement settings (keep fixed across the study):
      * NO response cache — a cache would replay identical answers and fake Det-var = 0.
      * temperature 0.0 (most favourable to RAGAS; variance at T=0 is a conservative finding),
        top_p 1.0, both sent explicitly. max_tokens 4096: ragas' default 1024 truncated
        GLM-5.3-Flash (a reasoning model) on 2/45 pilot claims (IncompleteOutputException).
      * context = the cited span (same input as the other Block 5 scorers); "all_docs"
        joins every cited_doc instead.
      * provider = groq | fireworks (same gpt-oss-20b weights); recorded in `config`.
      * Per claim: tokens / calls / HTTP requests in `last_usage` (retries = requests - calls),
        totals in `total_usage`; `last_status` = ok | skipped_no_source | judge_nan.

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
        top_p: float | None = 1.0,
        seed: int | None = None,
        max_tokens: int | None = 4096,
        price_in_per_mtok: float | None = None,
        price_out_per_mtok: float | None = None,
        context: str = "cited_span",
        provider: str = "groq",
        max_retries: int = 6,
        llm: Any = None,
        client: Any = None,
    ) -> None:
        if context not in ("cited_span", "all_docs"):
            raise ValueError("context must be 'cited_span' or 'all_docs'")
        prov = _provider(provider)
        self.provider = provider
        self.base_url = prov["base_url"]
        self.model = model or prov["model"]
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.context = context
        self.question: str | None = None
        self.n_fallback_question = 0
        self.n_nan = 0
        self.last_status = "ok"
        self._usage = _Usage()
        self._calls: list[dict] = []  # per LLM call of the current claim (judge audit log)
        self._http: list[dict] = []   # per HTTP attempt of the current claim
        self._loop = asyncio.new_event_loop()  # one persistent loop: AsyncOpenAI must not hop loops
        _warn_prices("RagasScorer", price_in_per_mtok, price_out_per_mtok)

        import ragas
        from ragas.metrics.collections import Faithfulness
        self.ragas_version = ragas.__version__

        if llm is None:
            from ragas.llms import llm_factory

            if client is None:
                import httpx
                from openai import AsyncOpenAI

                async def on_request(_request: Any) -> None:
                    self._usage.add("http_requests", 1)

                async def on_response(response: Any) -> None:
                    # every HTTP attempt, including the ones the SDK retries (429 / 5xx)
                    self._http.append({"status": response.status_code,
                                       "ratelimit": {k: v for k, v in response.headers.items()
                                                     if "ratelimit" in k.lower() or k.lower() == "retry-after"}})

                client = AsyncOpenAI(api_key=_api_key(provider, api_key), base_url=self.base_url,
                                     max_retries=max_retries,
                                     http_client=httpx.AsyncClient(
                                         event_hooks={"request": [on_request], "response": [on_response]},
                                         timeout=120))
            self._wrap_usage(client)
            kwargs = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if seed is not None:
                kwargs["seed"] = seed
            if max_tokens is not None:  # ragas default 1024 truncates reasoning models (pilot 2026-11-27)
                kwargs["max_tokens"] = max_tokens
            llm = llm_factory(self.model, provider="openai", client=client, **kwargs)  # no cache
        self._metric = Faithfulness(llm=llm)

    # -- token accounting -----------------------------------------------------
    def _wrap_usage(self, client: Any) -> None:
        """Intercept chat.completions.create to read `usage` from every raw response."""
        completions = client.chat.completions
        orig = completions.create

        async def create(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            resp = await orig(*args, **kwargs)
            self._usage.add_response(resp)
            self._calls.append(_call_record(resp, kwargs, (time.perf_counter() - t0) * 1000))
            return resp

        completions.create = create

    @property
    def last_calls(self) -> list[dict]:
        """Audit log of the last claim's LLM calls: returned model id, request response_format,
        finish_reason, tokens (incl. reasoning), latency and the raw judge output (statements,
        verdicts and reasons), so verdict reasons can be checked against the source."""
        return self._calls

    @property
    def last_http(self) -> list[dict]:
        return self._http

    @property
    def last_usage(self) -> dict:
        return self._usage.last

    @property
    def total_usage(self) -> dict:
        return self._usage.total

    @property
    def last_retries(self) -> int:
        return self._usage.last_retries

    def _cost(self) -> float:
        return self._usage.cost(self._usage.last, self.price_in, self.price_out)

    # -- interface ------------------------------------------------------------
    def set_question(self, question: str | None) -> None:
        """Set the original question of the bundle being scored (RAGAS user_input)."""
        self.question = question

    @property
    def config(self) -> dict:
        """Settings to log next to results for reproducibility."""
        return {"method": self.name, "provider": self.provider, "base_url": self.base_url,
                "model": self.model, "temperature": self.temperature, "top_p": self.top_p,
                "seed": self.seed, "max_tokens": self.max_tokens, "context": self.context, "cache": None,
                "prompt_version": f"ragas=={self.ragas_version} Faithfulness (collections)",
                "price_in_per_mtok": self.price_in, "price_out_per_mtok": self.price_out}

    def score(self, claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult:
        t0 = time.perf_counter()
        cid = _claim_id(claim)
        self._usage.reset_last()
        self._calls, self._http = [], []

        if self.context == "all_docs":
            contexts = [d.get("text", "") for d in cited_docs or [] if d.get("text")]
        else:
            contexts = [claim.cited_span_text] if claim.cited_span_text else []

        if not cited_docs or claim.alignment == "none" or not contexts or not claim.claim_text:
            self.last_status = "skipped_no_source"
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
        self.last_status = "ok"
        if math.isnan(value):  # judge produced no statements -> treat as unsupported, but count it
            self.n_nan += 1
            self.last_status = "judge_nan"
            value = 0.0
        return ScorerResult(claim_id=cid, method=self.name,
                            score=max(0.0, min(1.0, value)),
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            cost_usd=self._cost(), deterministic=False)

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.close()


# ── SelfCheckGPTScorer ────────────────────────────────────────────────────────

SC_NLI_MODEL = "potsawee/deberta-v3-large-mnli"  # NLIConfig.nli_model in the official selfcheckgpt package

# RAGTruth QA generation prompt (verbatim template from RAGTruth source_info.jsonl `prompt`),
# so the stochastic samples are drawn under the same conditions as the response being checked.
RAGTRUTH_QA_PROMPT = (
    "Briefly answer the following question:\n{question}\n"
    "Bear in mind that your response should be strictly based on the following {n} passages:\n"
    "{passages}"
    "In case the passages do not contain the necessary information to answer the question, "
    "please reply with: \"Unable to answer based on given passages.\"\noutput:"
)
RAGTRUTH_QA_PROMPT_VERSION = "ragtruth-qa-v1"
_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


class SelfCheckGPTScorer:
    """Baseline 3 — SelfCheckGPT, NLI variant (Manakul et al., 2023, EMNLP).

    Idea: if the model really "knows" a fact, re-asking the same question gives answers that
    agree with it; a hallucinated sentence tends to be contradicted by fresh samples.
      1. sampling (the ONLY LLM stage): ask the generator LLM the same question N more times
         (temperature 1.0, top_p 1.0) with the verbatim RAGTruth QA prompt;
      2. NLI (local, no tokens): for each sample, P(contradiction | claim, sample) with
         DeBERTa-v3-large-MNLI, normalised over {entailment, contradiction} (the model's 2 labels);
      3. hallucination score = mean over samples;  our score = 1 - that  (1 = consistent).
    The NLI step re-implements `selfcheckgpt.SelfCheckNLI.predict` (MIT, v0.1.7): pair order
    (sentence, sample), softmax, probs[1] = contradiction, mean over samples.

    Samples are drawn once per response (same question + same passages) and shared by all its
    claims, as in the paper. Call `reset_samples()` between Det-var reruns so every rerun draws
    fresh samples — otherwise rerun variance would be faked to 0.

    Accounting, kept separate per stage:
      * generation: `last_usage` / `total_usage` (tokens, calls, HTTP requests -> retries);
        all N sampling calls of a response are charged to the claim that triggered them.
      * NLI: `last_nli` / `total_nli` (pairs, milliseconds); no API cost.
    `last_status` = ok | skipped_no_source | no_samples.

    Interface gap (same as RAGAS): the question is set per bundle via `set_question()`. Without it
    the prompt falls back to "(no question given)" and `n_fallback_question` is incremented.
    """

    name = "selfcheckgpt-nli"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        n_samples: int = 3,
        temperature: float = 1.0,
        top_p: float | None = 1.0,
        seed: int | None = None,
        reasoning_effort: str | None = "low",
        max_completion_tokens: int = 2048,
        price_in_per_mtok: float | None = None,
        price_out_per_mtok: float | None = None,
        nli_model: str = SC_NLI_MODEL,
        device: str | None = None,
        provider: str = "groq",
        max_retries: int = 6,
        client: Any = None,
        nli: Any = None,
    ) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        prov = _provider(provider)
        self.provider = provider
        self.base_url = prov["base_url"]
        self.model = model or prov["model"]
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.nli_model = nli_model
        self.question: str | None = None
        self.n_fallback_question = 0
        self.n_empty_samples = 0
        self.last_status = "ok"
        self._usage = _Usage()
        self.last_nli = {"pairs": 0, "ms": 0.0}
        self.total_nli = {"pairs": 0, "ms": 0.0}
        self._samples_cache: dict[tuple, list[str]] = {}
        _warn_prices("SelfCheckGPTScorer", price_in_per_mtok, price_out_per_mtok)

        if client is None:
            import httpx
            from openai import OpenAI

            def on_request(_request: Any) -> None:
                self._usage.add("http_requests", 1)

            client = OpenAI(api_key=_api_key(provider, api_key), base_url=self.base_url,
                            max_retries=max_retries,
                            http_client=httpx.Client(event_hooks={"request": [on_request]}, timeout=120))
        self._client = client

        # nli: callable(list[(claim, sample)]) -> list[P(contradiction)]; injectable for tests.
        self.nli_device = "injected"
        self._nli = nli if nli is not None else self._load_nli(nli_model, device)

    # -- NLI (DeBERTa-v3-large-MNLI, local, deterministic) ---------------------
    def _load_nli(self, model_name: str, device: str | None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_device = dev if dev == "cpu" else f"{dev}:{torch.cuda.get_device_name(0)}"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()

        @torch.no_grad()
        def contradiction_probs(pairs: list[tuple[str, str]]) -> list[float]:
            out = []
            for sentence, sample in pairs:  # one pair at a time, exactly like SelfCheckNLI.predict
                enc = tok([(sentence, sample)], add_special_tokens=True, padding="longest",
                          truncation=True, return_tensors="pt",
                          return_token_type_ids=True, return_attention_mask=True).to(dev)
                probs = torch.softmax(mdl(**enc).logits, dim=-1)
                out.append(float(probs[0][1].item()))  # prob(contradiction)
            return out

        return contradiction_probs

    # -- sampling -------------------------------------------------------------
    def _prompt(self, question: str, cited_docs: list[dict]) -> str:
        texts = [d.get("text", "") for d in cited_docs]
        n = len(texts)
        passages = "".join(f"passage {i}:{t}\n\n" for i, t in enumerate(texts, 1))
        return RAGTRUTH_QA_PROMPT.format(
            question=question, n=_NUM_WORDS[n] if n < len(_NUM_WORDS) else str(n), passages=passages)

    def _draw_samples(self, prompt: str) -> list[str]:
        samples = []
        extra: dict[str, Any] = {"max_completion_tokens": self.max_completion_tokens}
        if self.top_p is not None:
            extra["top_p"] = self.top_p
        if self.seed is not None:
            extra["seed"] = self.seed
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        for _ in range(self.n_samples):
            resp = self._client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature, **extra)
            self._usage.add_response(resp)
            text = (resp.choices[0].message.content or "").strip()
            if text:
                samples.append(text)
            else:  # e.g. reasoning used up the token budget; drop it but count it
                self.n_empty_samples += 1
        return samples

    @property
    def last_usage(self) -> dict:
        return self._usage.last

    @property
    def total_usage(self) -> dict:
        return self._usage.total

    @property
    def last_retries(self) -> int:
        return self._usage.last_retries

    # -- interface ------------------------------------------------------------
    def set_question(self, question: str | None) -> None:
        """Set the original question of the bundle being scored."""
        self.question = question

    def reset_samples(self) -> None:
        """Forget drawn samples; call between Det-var reruns so each rerun resamples."""
        self._samples_cache.clear()

    @property
    def config(self) -> dict:
        return {"method": self.name, "provider": self.provider, "base_url": self.base_url,
                "model": self.model, "generator_model": self.model, "n_samples": self.n_samples,
                "sampling_unit": "response", "temperature": self.temperature, "top_p": self.top_p,
                "seed": self.seed, "reasoning_effort": self.reasoning_effort,
                "max_completion_tokens": self.max_completion_tokens, "nli_model": self.nli_model,
                "nli_device": self.nli_device,
                "prompt_version": RAGTRUTH_QA_PROMPT_VERSION,
                "price_in_per_mtok": self.price_in, "price_out_per_mtok": self.price_out}

    def score(self, claim: AlignedClaim, cited_docs: list[dict]) -> ScorerResult:
        t0 = time.perf_counter()
        cid = _claim_id(claim)
        self._usage.reset_last()
        self.last_nli = {"pairs": 0, "ms": 0.0}

        if not cited_docs or claim.alignment == "none" or not claim.claim_text:
            self.last_status = "skipped_no_source"
            return ScorerResult(claim_id=cid, method=self.name, score=0.0,
                                latency_ms=(time.perf_counter() - t0) * 1000,
                                cost_usd=0.0, deterministic=False)

        question = self.question
        if not question:
            question = "(no question given)"
            self.n_fallback_question += 1

        key = (question, tuple(d.get("text", "") for d in cited_docs))
        if key not in self._samples_cache:
            self._samples_cache[key] = self._draw_samples(self._prompt(question, cited_docs))
        samples = self._samples_cache[key]
        cost = self._usage.cost(self._usage.last, self.price_in, self.price_out)

        if not samples:  # every sample came back empty -> nothing to compare against
            self.last_status = "no_samples"
            value = 0.0
        else:
            self.last_status = "ok"
            t_nli = time.perf_counter()
            contra = self._nli([(claim.claim_text, s) for s in samples])
            ms = (time.perf_counter() - t_nli) * 1000
            self.last_nli = {"pairs": len(samples), "ms": ms}
            self.total_nli["pairs"] += len(samples)
            self.total_nli["ms"] += ms
            value = 1.0 - sum(contra) / len(contra)
        return ScorerResult(claim_id=cid, method=self.name,
                            score=max(0.0, min(1.0, value)),
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            cost_usd=cost, deterministic=False)
