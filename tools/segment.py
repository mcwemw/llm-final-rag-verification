"""Deterministic claim segmenter — shared spec for Block 3 (agreed with the Gate owner, 2026-11-24).

    segment(answer) -> list[Piece]

Rules (v1):
  1. Boundaries: sentence end (. ! ? followed by whitespace), newlines, ';',
     and ', while|but|whereas|although' (NOT 'and': it over-splits coordinated NPs).
     Sentence ends after common abbreviations (e.g., i.e., Dr., U.S., ...) are not boundaries.
  2. Leading connective / bullet / list markers are stripped from a piece
     ("while X" -> "X", "- X" -> "X", "1. X" -> "X", "Passage 2: X" -> "X").
     A piece that is only a marker ("1.", "(Passage 2)") is not a claim and is skipped.
  3. A source-attribution prefix is stripped, the content is kept:
     "Based on the provided passages, X" / "According to passage 3, X" / "The passage states that X" -> "X".
  4. A piece ending in ':' followed by a short piece (< MIN_WORDS words, e.g. a phone number on the
     next line) is merged with it, so "The phone number is:\\n555-1234" stays one checkable claim.
  5. Excluded (kept in the output with `excluded=True`, never deleted):
       - "meta":  content-free preambles ending in ':' ("Based on the passages, the answer is:")
                  and pleasantries ("Sure!", "I hope this helps!", "Let me know ...")
       - "short": fewer than MIN_WORDS words.
Invariant for every piece: answer[char_start:char_end] == text.
"""
import re
from dataclasses import dataclass, asdict
from typing import Optional

RULES_VERSION = "1.0"
MIN_WORDS = 4

# sentence end | newline (bullet lines) | semicolon | ", while/but/whereas/although" (connective stripped later)
BOUNDARY = re.compile(r"(?<=[.!?])\s+|\s*\n\s*|\s*;\s*|,\s+(?=(?:while|but|whereas|although)\b)", re.I)

ABBREV = re.compile(
    r"(?:\b(?:e\.g|i\.e|etc|vs|approx|dr|mr|mrs|ms|st|no|fig|inc|ltd|jr|sr|"
    r"tbsp|tsp|oz|lbs?|ft|sq|gal|qt|pt|mins?|hrs?|sec|mo|yr|yrs|est|dept|ave|blvd|mt)\.|\b(?:[A-Z]\.){2,})$",
    re.I)

LEAD_STRIP = re.compile(
    r"^(?:"
    r"(?:while|but|whereas|although)\s+"                                     # connective
    r"|[-*•·]\s+"                                                            # bullet
    r"|\(?\d{1,2}[.)]\s+"                                                    # numbered list
    r"|\(passage\s*\d+\)\s*[:.,-]?\s*|passage\s*\d+\s*[:.-]\s*"              # "(Passage 2)" / "Passage 2:" (NOT "Passage 2 says")
    r")+",
    re.I,
)
MARKER_ONLY = re.compile(r"^(?:[-*•·]|\(?\d{1,2}[.):]?\)?|\(?passage\s*\d+\)?[:.]?)$", re.I)

ATTRIBUTION = re.compile(
    r"^(?:based on|according to|as (?:stated|mentioned) in)\s+(?:the\s+)?(?:given\s+|provided\s+)?"
    r"(?:passages?|context|information)(?:\s+\d+)?(?:\s+provided)?\s*,\s*"
    r"|^(?:the\s+)?(?:given\s+|provided\s+)?passages?(?:\s+\d+)?\s+"
    r"(?:states?|says?|mentions?|indicates?|explains?|notes?|suggests?|recommends?|shows?|reports?)\s+(?:that\s+)?",
    re.I,
)
PREAMBLE = re.compile(r"^(?:based on|according to|here (?:is|are)|therefore|in summary|to answer|so)\b.*:$", re.I | re.S)
PLEASANTRY = re.compile(
    r"^(?:sure|certainly|of course|great question|i hope this helps|hope this helps|let me know|feel free)\b", re.I
)


@dataclass
class Piece:
    text: str
    char_start: int
    char_end: int
    excluded: bool = False
    exclude_reason: Optional[str] = None  # "meta" | "short" | None

    def to_dict(self):
        return asdict(self)


def _raw_pieces(answer):
    """Split on boundaries, return [(start, end)] with whitespace trimmed."""
    cuts, pos = [], 0
    for m in BOUNDARY.finditer(answer):
        # do not cut after an abbreviation like "e.g." or "U.S."
        if m.group(0).strip() == "" and "\n" not in m.group(0) and ABBREV.search(answer[pos:m.start()]):
            continue
        cuts.append((pos, m.start()))
        pos = m.end()
    cuts.append((pos, len(answer)))
    out = []
    for s, e in cuts:
        seg = answer[s:e]
        s2, e2 = s + (len(seg) - len(seg.lstrip())), e - (len(seg) - len(seg.rstrip()))
        if e2 > s2:
            out.append((s2, e2))
    return out


def _strip(answer, s, e, rx):
    # match on the substring: '^' does not anchor at a `pos` offset in re.match(answer, s, e)
    m = rx.match(answer[s:e])
    return (s + m.end(), e) if m and s + m.end() < e else (s, e)


def _n_words(t):
    return len(t.split())


def segment(answer: str):
    pieces = []
    for s, e in _raw_pieces(answer):
        if MARKER_ONLY.match(answer[s:e]):
            continue
        s, e = _strip(answer, s, e, LEAD_STRIP)
        if MARKER_ONLY.match(answer[s:e]):
            continue
        pieces.append([s, e])

    # rule 4: "...is:" + short follow-up -> one claim
    merged = []
    for s, e in pieces:
        if merged and answer[merged[-1][1] - 1] == ":" and _n_words(answer[s:e]) < MIN_WORDS:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out = []
    for s, e in merged:
        text = answer[s:e]
        if PLEASANTRY.match(text) or PREAMBLE.match(text):
            out.append(Piece(text, s, e, True, "meta"))
            continue
        s, e = _strip(answer, s, e, ATTRIBUTION)
        text = answer[s:e]
        if _n_words(text) < MIN_WORDS:
            out.append(Piece(text, s, e, True, "short"))
        else:
            out.append(Piece(text, s, e))
    for p in out:
        assert answer[p.char_start:p.char_end] == p.text
    return out
