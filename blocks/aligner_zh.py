"""Block 4 — Claim <-> Source Aligner, Chinese path (--lang zh). PROPOSAL, not locked.

Deterministic, no external deps. Returns the same AlignedClaim as blocks/aligner.py.

  explicit : the claim's own raw text carries a citation marker (【doc_016】 etc.) whose doc_id is
             among the bundle's citations. Several markers -> the one with the highest bigram
             Jaccard. This matches the contract's meaning of "explicit" ("claim falls inside a
             declared citation"); the English path approximates it by substring match because
             RAGTruth answers have no inline markers.
  nearest  : no usable marker -> citation with the highest character-bigram Jaccard
             (bigrams of CJK characters + lower-cased Latin words / numbers),
             if >= NEAREST_THRESHOLD_ZH. The English word Jaccard (\\w+) treats a whole Chinese
             sentence as one "word", so it never overlaps.
  none     : otherwise -> UNSOURCED downstream.
"""
from __future__ import annotations

import re
from typing import Any

from blocks.aligner import AlignedClaim
from blocks.segmenter import Claim
from blocks.segmenter_zh import _MARKER, clean, segment_zh

NEAREST_THRESHOLD_ZH = 0.05  # same value as the English path; see proposal for the measured distribution

_DOC_ID = re.compile(r"doc_[A-Za-z0-9_-]+")
_CJK_RUN = re.compile(r"[㐀-鿿]+")
_LATIN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")


def grams(text: str) -> set[str]:
    """Character bigrams inside CJK runs (single char if the run has length 1) + Latin/number tokens."""
    out: set[str] = set()
    for run in _CJK_RUN.findall(text):
        out.update(run[i:i + 2] for i in range(len(run) - 1)) if len(run) > 1 else out.add(run)
    out.update(t.lower() for t in _LATIN.findall(_CJK_RUN.sub(" ", text)))
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def markers(raw: str) -> list[str]:
    """doc_ids cited by markers inside a raw claim span, first-appearance order."""
    seen: list[str] = []
    for m in _MARKER.finditer(raw):
        for d in _DOC_ID.findall(m.group()):
            if d not in seen:
                seen.append(d)
    return seen


def align_zh(claim: Claim, citations: list[dict[str, Any]], answer: str) -> AlignedClaim:
    def result(doc: dict | None, alignment: str) -> AlignedClaim:
        return AlignedClaim(claim.claim_text, claim.char_start, claim.char_end,
                            doc.get("doc_id") if doc else None, doc.get("text") if doc else None,
                            alignment)

    if not citations:
        return result(None, "none")
    g = grams(claim.claim_text)
    by_id = {c.get("doc_id"): c for c in citations}
    marked = [by_id[d] for d in markers(answer[claim.char_start:claim.char_end]) if d in by_id]
    if marked:
        return result(max(marked, key=lambda c: jaccard(g, grams(c.get("text", "")))), "explicit")
    best = max(citations, key=lambda c: jaccard(g, grams(c.get("text", ""))))
    if jaccard(g, grams(best.get("text", ""))) >= NEAREST_THRESHOLD_ZH:
        return result(best, "nearest")
    return result(None, "none")


def align_bundle_zh(answer_bundle: dict[str, Any]) -> list[AlignedClaim]:
    answer: str = answer_bundle.get("answer", "")
    citations: list[dict] = answer_bundle.get("citations", [])
    return [align_zh(c, citations, answer) for c in segment_zh(answer)]


__all__ = ["align_zh", "align_bundle_zh", "grams", "jaccard", "markers", "clean"]
