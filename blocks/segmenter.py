"""Block 3 — Claim Segmenter.

Deterministic rule-based segmenter. No external deps.
Rule set locked 2026-11-24 with M-Survey/Noa.
2026-11-27: added language-aware strategy (Jimmy #7, --lang zh). `lang="en"` (default)
preserves the original behaviour exactly; zh/auto add Chinese sentence boundaries
(。！？；) and a CJK-aware length rule (so Chinese claims are no longer dropped for
having zero whitespace-words). Processing strategy only — Claim contract unchanged.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from blocks.lang import get_strategy

_CONNECTIVE = re.compile(r"(?<!\w)(while|but|whereas|although)(?!\w)", re.IGNORECASE)
_META = re.compile(
    r"^(based on (the |this )?(provided |given )?(passage|context|information|documents?)|"
    r"according to (the |this )?(provided |given )?(passage|context|information|documents?)|"
    r"(the |this )?(passage|context|document) (states?|says?|mentions?|indicates?|shows?|reports?))",
    re.IGNORECASE,
)
_TRAILING_PUNCT = re.compile(r"[.!?,;。！？；]+$")   # + Chinese enders


@dataclass(frozen=True)
class Claim:
    claim_text: str
    char_start: int
    char_end: int


def _keep(text: str, strat) -> bool:
    if not strat.long_enough(text):
        return False
    if _META.match(text):
        return False
    return True


def _boundary_split(answer: str, strat) -> list[tuple[str, int]]:
    """Return (chunk, start_offset) pairs split on sentence-ends, newlines, semicolons
    (and, for zh/auto, Chinese enders)."""
    pieces: list[tuple[str, int]] = []
    pos = 0
    for m in strat.boundary.finditer(answer):
        if m.start() == m.end() and m.start() == pos:
            continue  # zero-width match at current pos -> no progress, skip
        chunk = answer[pos:m.start()]
        if chunk.strip():
            pieces.append((chunk.strip(), pos))
        pos = m.end()
    tail = answer[pos:]
    if tail.strip():
        pieces.append((tail.strip(), pos))
    return pieces


def segment(answer: str, lang: str = "en") -> list[Claim]:
    if not answer or not answer.strip():
        return []
    strat = get_strategy(lang)

    raw_chunks = _boundary_split(answer, strat)
    claims: list[Claim] = []

    for chunk, chunk_offset in raw_chunks:
        # strip leading bullet markers
        bullet_match = re.match(r"^[-*•]\s*", chunk)
        if bullet_match:
            chunk_offset += bullet_match.end()
            chunk = chunk[bullet_match.end():]

        # connective split (English connectives; harmless on Chinese text)
        sub_chunks: list[tuple[str, int]] = []
        pos = 0
        for m in _CONNECTIVE.finditer(chunk):
            before = chunk[pos:m.start()]
            if before.strip():
                sub_chunks.append((before.strip(), chunk_offset + pos))
            sub_chunks.append((chunk[m.start():].split(None, 1)[0], chunk_offset + m.start()))
            pos = m.start() + len(m.group())
        rest = chunk[pos:]
        if rest.strip():
            sub_chunks.append((rest.strip(), chunk_offset + pos))

        if not sub_chunks:
            sub_chunks = [(chunk, chunk_offset)]

        for text, start in sub_chunks:
            clean = _TRAILING_PUNCT.sub("", text).strip()
            if _keep(clean, strat):
                end = start + len(text)
                claims.append(Claim(claim_text=clean, char_start=start, char_end=end))

    return claims


if __name__ == "__main__":   # CLI: python -m blocks.segmenter --lang zh "text"  (or stdin)
    import argparse, sys
    ap = argparse.ArgumentParser(description="Claim segmenter (--lang en|zh|auto)")
    ap.add_argument("text", nargs="?", help="answer text; omitted -> read stdin")
    ap.add_argument("--lang", default="en", choices=["en", "zh", "auto"])
    a = ap.parse_args()
    txt = a.text if a.text is not None else sys.stdin.read()
    cs = segment(txt, lang=a.lang)
    print(f"lang={a.lang}  claims={len(cs)}")
    for i, c in enumerate(cs):
        print(f"  [{i}] ({c.char_start},{c.char_end}) {c.claim_text}")
