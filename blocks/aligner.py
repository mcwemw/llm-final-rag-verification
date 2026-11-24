"""Block 4 — Claim ↔ Source Aligner.

Deterministic. No external deps.
Rule set locked 2026-11-24.

alignment values
  "explicit" — claim text is a substring of a cited span text
  "nearest"  — highest word-overlap citation (Jaccard on word sets)
  "none"     — no citation with overlap > NEAREST_THRESHOLD → UNSOURCED
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from blocks.segmenter import Claim

NEAREST_THRESHOLD = 0.05  # min Jaccard to assign "nearest"; below → "none"

_WORD = re.compile(r"\b\w+\b")


@dataclass(frozen=True)
class AlignedClaim:
    claim_text: str
    char_start: int
    char_end: int
    cited_doc_id: str | None
    cited_span_text: str | None
    alignment: str  # "explicit" | "nearest" | "none"


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def align(claim: Claim, citations: list[dict[str, Any]]) -> AlignedClaim:
    """Align one claim to the best-matching citation.

    citations — list of dicts with keys: doc_id, text (and optionally span_start/span_end)
    """
    if not citations:
        return AlignedClaim(
            claim_text=claim.claim_text,
            char_start=claim.char_start,
            char_end=claim.char_end,
            cited_doc_id=None,
            cited_span_text=None,
            alignment="none",
        )

    claim_lower = claim.claim_text.lower()
    claim_words = _words(claim.claim_text)

    best_doc_id: str | None = None
    best_text: str | None = None
    best_score: float = -1.0
    best_alignment = "none"

    for cit in citations:
        doc_id = cit.get("doc_id", "")
        span_text: str = cit.get("text", "")

        # explicit: exact substring match
        if claim_lower in span_text.lower():
            return AlignedClaim(
                claim_text=claim.claim_text,
                char_start=claim.char_start,
                char_end=claim.char_end,
                cited_doc_id=doc_id,
                cited_span_text=span_text,
                alignment="explicit",
            )

        score = _jaccard(claim_words, _words(span_text))
        if score > best_score:
            best_score = score
            best_doc_id = doc_id
            best_text = span_text

    if best_score >= NEAREST_THRESHOLD:
        best_alignment = "nearest"
    else:
        best_doc_id = None
        best_text = None

    return AlignedClaim(
        claim_text=claim.claim_text,
        char_start=claim.char_start,
        char_end=claim.char_end,
        cited_doc_id=best_doc_id,
        cited_span_text=best_text,
        alignment=best_alignment,
    )


def align_bundle(answer_bundle: dict[str, Any]) -> list[AlignedClaim]:
    """Segment an AnswerBundle and align all claims.

    answer_bundle keys: answer, citations[{doc_id, text, span_start, span_end}]
    """
    from blocks.segmenter import segment

    answer: str = answer_bundle.get("answer", "")
    citations: list[dict] = answer_bundle.get("citations", [])

    claims = segment(answer)
    return [align(c, citations) for c in claims]
