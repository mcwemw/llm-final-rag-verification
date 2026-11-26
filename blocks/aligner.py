"""Block 4 — Claim ↔ Source Aligner.

Deterministic. No external deps.
Rule set locked 2026-11-24.
2026-11-27: language-aware tokens (Jimmy #7). `lang="en"` (default) is unchanged.

alignment values
  "explicit" — claim text is a substring of a cited span text
  "nearest"  — highest word-overlap citation (Jaccard on token sets)
  "none"     — no citation with overlap > NEAREST_THRESHOLD → UNSOURCED

🔴 Known limitation (zh/auto): Chinese alignment uses LEXICAL char-bigram Jaccard
(blocks.lang.ZH_ALIGNMENT_METHOD). It is NOT semantic entailment — a Chinese claim that
is a paraphrase of its evidence (no shared characters) may still be "none". Semantic zh
alignment would require a Chinese NLI model and is out of scope for this slice. We surface
this honestly rather than pretend Chinese semantic alignment is solved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blocks.segmenter import Claim, segment
from blocks.lang import get_strategy, ZH_ALIGNMENT_METHOD

NEAREST_THRESHOLD = 0.05  # min Jaccard to assign "nearest"; below → "none"


@dataclass(frozen=True)
class AlignedClaim:
    claim_text: str
    char_start: int
    char_end: int
    cited_doc_id: str | None
    cited_span_text: str | None
    alignment: str  # "explicit" | "nearest" | "none"


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def align(claim: Claim, citations: list[dict[str, Any]], lang: str = "en") -> AlignedClaim:
    """Align one claim to the best-matching citation.

    citations — list of dicts with keys: doc_id, text (and optionally span_start/span_end)
    lang — en (default) | zh | auto. Only changes tokenization for the overlap score;
           the AlignedClaim contract is unchanged.
    """
    strat = get_strategy(lang)

    if not citations:
        return AlignedClaim(claim.claim_text, claim.char_start, claim.char_end,
                            None, None, "none")

    claim_lower = claim.claim_text.lower()
    claim_tokens = strat.tokens(claim.claim_text)

    best_doc_id: str | None = None
    best_text: str | None = None
    best_score: float = -1.0

    for cit in citations:
        doc_id = cit.get("doc_id", "")
        span_text: str = cit.get("text", "")

        if claim_lower in span_text.lower():          # explicit substring (works for zh too)
            return AlignedClaim(claim.claim_text, claim.char_start, claim.char_end,
                                doc_id, span_text, "explicit")

        score = _jaccard(claim_tokens, strat.tokens(span_text))
        if score > best_score:
            best_score = score
            best_doc_id = doc_id
            best_text = span_text

    if best_score >= NEAREST_THRESHOLD:
        alignment = "nearest"
    else:
        alignment = "none"
        best_doc_id = None
        best_text = None

    return AlignedClaim(claim.claim_text, claim.char_start, claim.char_end,
                        best_doc_id, best_text, alignment)


def align_bundle(answer_bundle: dict[str, Any], lang: str = "en") -> list[AlignedClaim]:
    answer = answer_bundle.get("answer", "")
    citations = answer_bundle.get("citations", [])
    claims = segment(answer, lang=lang)
    return [align(c, citations, lang=lang) for c in claims]


if __name__ == "__main__":   # CLI demo: python -m blocks.aligner --lang zh
    import argparse, json, sys
    ap = argparse.ArgumentParser(description="Claim aligner (--lang en|zh|auto)")
    ap.add_argument("bundle_json", nargs="?", help="AnswerBundle json path; omitted -> stdin")
    ap.add_argument("--lang", default="en", choices=["en", "zh", "auto"])
    a = ap.parse_args()
    raw = open(a.bundle_json).read() if a.bundle_json else sys.stdin.read()
    results = align_bundle(json.loads(raw), lang=a.lang)
    if a.lang in ("zh", "auto"):
        print(f"[zh note] alignment = {ZH_ALIGNMENT_METHOD}")
    print(f"lang={a.lang}  claims={len(results)}")
    for i, r in enumerate(results):
        print(f"  [{i}] {r.alignment:8s} doc={r.cited_doc_id} :: {r.claim_text}")
