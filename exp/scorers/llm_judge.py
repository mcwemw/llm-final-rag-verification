"""LLM-judge faithfulness scorer (RAGAS-style).

Calls Groq (no-train, IP-safe for public RAGTruth data) with the RAGAS
faithfulness decomposition prompt.  Returns a score in [0, 1] per response.

Non-deterministic by design (temperature > 0):
  same input, different run -> different score -> determinism_var > 0.
  This is the key measurement the harness reports for this scorer.

Usage:
    from scorers.llm_judge import score_batch
    scores = score_batch(records, model="llama-3.3-70b-versatile", n_runs=3)
"""
from __future__ import annotations
import os
import json
import re
import time
from typing import Optional

# --- RAGAS-style decomposition prompt ---
_SYS = """\
You are a faithful passage verifier.
Given a question, a retrieved context, and a generated answer,
break the answer into atomic claims and judge each claim:
  SUPPORTED   - directly supported by the context
  CONTRADICTS - contradicts the context
  UNSUPPORTED - not mentioned in the context

Respond with a JSON object only, no markdown:
{"claims": [{"text": "...", "verdict": "SUPPORTED|CONTRADICTS|UNSUPPORTED"}],
 "faithfulness_score": <float 0.0-1.0 = supported / total claims>}
"""

_USER = """\
Question: {question}

Context:
{context}

Answer:
{answer}
"""


def _groq_client():
    """Lazy Groq client. Reads GROQ_API_KEY from env (source ~/.zsh_secrets)."""
    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError("pip install groq") from e
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set. Run: source ~/.zsh_secrets")
    return Groq(api_key=key)


def _parse_score(text: str) -> Optional[float]:
    """Extract faithfulness_score from LLM response (robust to extra prose)."""
    # try strict JSON first
    try:
        obj = json.loads(text)
        return float(obj["faithfulness_score"])
    except Exception:
        pass
    # regex fallback
    m = re.search(r'"faithfulness_score"\s*:\s*([0-9.]+)', text)
    if m:
        return float(m.group(1))
    return None


def score_one(rec: dict, client, model: str, temperature: float = 0.3) -> Optional[float]:
    """Score a single record. Returns None on parse failure."""
    q = rec.get("question", "")
    if not q:
        # RAGTruth format: question embedded in source
        src = rec.get("source_text", "")
        q = src[:200] if src else "(no question)"
    ctx = rec.get("source_text", str(rec.get("passages", "")))[:3000]
    ans = rec.get("response", rec.get("response_text", ""))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": _USER.format(
                    question=q, context=ctx, answer=ans)},
            ],
            temperature=temperature,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content
        return _parse_score(raw)
    except Exception as e:
        print(f"  [warn] LLM call failed: {e}")
        return None


def score_batch(
    records: list[dict],
    model: str = "llama-3.3-70b-versatile",
    n_runs: int = 1,
    temperature: float = 0.3,
    delay_sec: float = 0.3,
    limit: int = 0,
) -> list[list[Optional[float]]]:
    """
    Returns scores[run][record_idx].
    n_runs > 1: same records scored n_runs times to measure determinism.
    limit: truncate records to first N (0 = all).
    """
    client = _groq_client()
    recs = records[:limit] if limit else records
    all_runs = []
    for run_i in range(n_runs):
        run_scores = []
        for j, rec in enumerate(recs):
            s = score_one(rec, client, model, temperature)
            run_scores.append(s)
            if delay_sec and j < len(recs) - 1:
                time.sleep(delay_sec)
        all_runs.append(run_scores)
        if run_i < n_runs - 1:
            print(f"  run {run_i+1}/{n_runs} done, sleeping 2s…")
            time.sleep(2)
    return all_runs


def threshold_scores(scores: list[Optional[float]], t: float = 0.6) -> list[int]:
    """Binary verdict: score < t (or None) → flagged (hallucinated=1)."""
    return [1 if (s is None or s < t) else 0 for s in scores]
