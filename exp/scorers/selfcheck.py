"""SelfCheckGPT-NLI faithfulness scorer (sampling consistency).

Published technique (Manakul et al., 2023): a faithful claim is *consistent*
across stochastically sampled regenerations; a hallucinated claim is not.

Operationalization for this RAG eval (matches the harness interface):
  1. Regenerate N stochastic answers from the same (question, passages) prompt
     via Groq (temperature high).  No-train, IP-safe for public RAGTruth data.
  2. Split the ORIGINAL response into sentences; for each sentence, average the
     NLI entailment probability (premise = sample, hypothesis = sentence) over
     the N samples.
  3. response consistency = mean over sentences, in [0, 1].
     High = consistent across samples = faithful.  pred = score < threshold → flag.

Non-deterministic by design (sampling) → determinism_var > 0 (a measured axis,
reported alongside RAGAS; this is the point, not a bug).

Usage (from run_harness.py):
    from scorers.selfcheck import score_batch
    all_runs = score_batch(records, model="llama-3.3-70b-versatile",
                           n_samples=3, n_runs=2, temperature=1.0)
    # all_runs: list[n_runs] of list[n_records] consistency scores in [0,1]
"""
from __future__ import annotations
import os, re, json
from typing import Optional

# reuse the project's Groq client + NLI pipeline (same models as RAGAS / NLI scorer)
from scorers.llm_judge import _groq_client
from scorers.nli_scorer import _pipeline, _entail_prob

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_SYS = "You answer the question strictly using only the provided passages. Be concise; no preamble."
_USER = "Passages:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"

_QMAP: Optional[dict] = None


def _questions() -> dict:
    """source_id -> question, from RAGTruth source_info.jsonl (QA). {} if absent."""
    global _QMAP
    if _QMAP is not None:
        return _QMAP
    _QMAP = {}
    p = os.path.join(_DATA, "source_info.jsonl")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    s = json.loads(line)
                    si = s.get("source_info", {})
                    if isinstance(si, dict) and "question" in si:
                        _QMAP[str(s.get("source_id"))] = si["question"]
                except Exception:
                    continue
    return _QMAP


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if len(s.strip()) > 3]


def _gen_samples(rec: dict, client, model: str, n: int, temperature: float) -> list[str]:
    q = _questions().get(str(rec.get("source_id")), "") or rec.get("question", "")
    if not q:
        q = "Answer the question implied by the passages, using only the passages."
    ctx = rec.get("source_text", str(rec.get("passages", "")))[:3000]
    user = _USER.format(ctx=ctx, q=q)
    out = []
    for _ in range(n):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
                temperature=temperature, max_tokens=256)
            out.append(r.choices[0].message.content or "")
        except Exception as e:
            print(f"  [selfcheck warn] sample gen failed: {e}")
    return out


def _consistency(response: str, samples: list[str], pipe) -> float:
    sents = _sentences(response)
    if not sents or not samples:
        return 0.5
    pairs = [{"text": smp[:512], "text_pair": s[:256]} for s in sents for smp in samples]
    results = pipe(pairs, truncation=True, max_length=512)
    probs = [_entail_prob(r if isinstance(r, list) else [r]) for r in results]
    # reshape -> mean over samples per sentence, then mean over sentences
    k = len(samples)
    per_sent = [sum(probs[i * k:(i + 1) * k]) / k for i in range(len(sents))]
    return float(sum(per_sent) / len(per_sent))


def score_batch(records: list[dict], model: str = "llama-3.3-70b-versatile",
                n_samples: int = 3, n_runs: int = 2, temperature: float = 1.0) -> list:
    """Returns all_runs: list[n_runs] of list[n_records] consistency scores [0,1]."""
    client = _groq_client()
    pipe = _pipeline()
    all_runs = []
    for run in range(n_runs):
        scores = []
        for j, rec in enumerate(records):
            samples = _gen_samples(rec, client, model, n_samples, temperature)
            scores.append(_consistency(rec.get("response", ""), samples, pipe))
            if (j + 1) % 20 == 0:
                print(f"  [selfcheck] run {run + 1}/{n_runs}: {j + 1}/{len(records)}")
        all_runs.append(scores)
    return all_runs


def threshold_scores(scores: list[float], t: float = 0.5) -> list[int]:
    """consistency < t → flagged (hallucinated=1), same convention as nli/llm."""
    return [1 if s < t else 0 for s in scores]
