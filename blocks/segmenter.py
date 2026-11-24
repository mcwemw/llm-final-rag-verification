"""Block 3 — Claim Segmenter.

Deterministic rule-based segmenter. No external deps.
Rule set locked 2026-11-24 with M-Survey/Noa.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|[\n\r]+|;\s*", re.MULTILINE)
_CONNECTIVE = re.compile(r"(?<!\w)(while|but|whereas|although)(?!\w)", re.IGNORECASE)
_META = re.compile(
    r"^(based on (the |this )?(provided |given )?(passage|context|information|documents?)|"
    r"according to (the |this )?(provided |given )?(passage|context|information|documents?)|"
    r"(the |this )?(passage|context|document) (states?|says?|mentions?|indicates?|shows?|reports?))",
    re.IGNORECASE,
)
_TRAILING_PUNCT = re.compile(r"[.!?,;]+$")
_MIN_WORDS = 4


@dataclass(frozen=True)
class Claim:
    claim_text: str
    char_start: int
    char_end: int


def _keep(text: str) -> bool:
    if len(text.split()) < _MIN_WORDS:
        return False
    if _META.match(text):
        return False
    return True


def _boundary_split(answer: str) -> list[tuple[str, int]]:
    """Return (chunk, start_offset) pairs split on sentence-ends, newlines, semicolons."""
    pieces: list[tuple[str, int]] = []
    pos = 0
    for m in _BOUNDARY.finditer(answer):
        chunk = answer[pos : m.start()]
        if chunk.strip():
            pieces.append((chunk.strip(), pos))
        pos = m.end()
    tail = answer[pos:]
    if tail.strip():
        pieces.append((tail.strip(), pos))
    return pieces


def segment(answer: str) -> list[Claim]:
    if not answer or not answer.strip():
        return []

    raw_chunks = _boundary_split(answer)
    claims: list[Claim] = []

    for chunk, chunk_offset in raw_chunks:
        # strip leading bullet markers
        bullet_match = re.match(r"^[-*•]\s*", chunk)
        if bullet_match:
            chunk_offset += bullet_match.end()
            chunk = chunk[bullet_match.end():]

        # connective split
        sub_chunks: list[tuple[str, int]] = []
        pos = 0
        for m in _CONNECTIVE.finditer(chunk):
            before = chunk[pos : m.start()]
            if before.strip():
                sub_chunks.append((before.strip(), chunk_offset + pos))
            sub_chunks.append((chunk[m.start():].split(None, 1)[0], chunk_offset + m.start()))
            pos = m.start() + len(m.group())
        rest = chunk[pos:]
        if rest.strip():
            sub_chunks.append((rest.strip(), chunk_offset + pos))

        if not sub_chunks:
            sub_chunks = [(chunk, chunk_offset)]

        # collect valid claims
        for text, start in sub_chunks:
            clean = _TRAILING_PUNCT.sub("", text).strip()
            if _keep(clean):
                end = start + len(text)
                claims.append(Claim(claim_text=clean, char_start=start, char_end=end))

    return claims
