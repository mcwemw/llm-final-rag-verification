"""Block 3 — Claim Segmenter, Chinese path (--lang zh). PROPOSAL, not locked.

Deterministic, rule-based, no external deps. Mirrors blocks/segmenter.py (English, locked
2026-11-24) but with rules that work on Chinese RAG answers (course Program 2 output):

  1. boundaries   : 。！？；!? and newlines; ASCII '.' only before whitespace/end (keeps "1.33").
  2. connectives  : split before a contrast connective after a comma — ，但是 / ，但 / ，然而 / ，不過 / ，而(not 而且)
                    (English path splits on while / but / whereas / although).
  3. leading junk : list markers (- * • 1. 1) (1) 一、), blockquote '>', heading '#'.
                    A piece that is only bold text ("**自行保存出勤證明**") is a heading -> skipped.
                    Concessive "雖然A，但B" is not split (A alone is not a claim).
  4. claim_text   : markdown (** __ ` *) and citation markers (【doc_016】 [doc_003, doc_010] 〔doc_006〕 （doc_011）)
                    removed, trailing punctuation stripped. char_start/char_end still cover the raw
                    text INCLUDING the markers, so the aligner can read them back.
  5. marker-only  : a piece that is only citation markers (e.g. "【doc_016】" after the 。) is merged
                    into the previous claim instead of being dropped.
  6. keep filter  : >= MIN_UNITS content units (1 CJK char = 1 unit, 1 Latin word / number = 1 unit;
                    English uses 4 whitespace words, which never fires on Chinese), and not meta-only
                    (e.g. "根據提供的文件：", "參考文件：請假權益（doc_011）").
"""
from __future__ import annotations

import re

from blocks.segmenter import Claim

MIN_UNITS = 6

_BOUNDARY = re.compile(r"[。！？；!?;]+|\.(?=\s|$)|\n+")
_CONNECTIVE = re.compile(r"[，,](?=但是|但|然而|不過|而(?!且))")
# no '^': used with pattern.match(answer, pos), which already anchors at pos
_LEADING = re.compile(
    r"(?:\s|>|#+|[-*•·]\s|\d+[.)、．]\s?|[（(]\d+[）)]|[一二三四五六七八九十]+、)+"
)
_CONCESSIVE = re.compile(r"(?:\*\*)?(?:雖然|雖|儘管|即使)")  # "雖然A，但B": keep whole
_HEADING = re.compile(r"\*\*[^*\n]+\*\*[：:]?")  # a line that is only bold text = heading
_DOC_IDS = r"doc_[A-Za-z0-9_-]+(?:\s*[,，、;；]\s*doc_[A-Za-z0-9_-]+)*"
_MARKER = re.compile(r"[\[【［〔（(]\s*" + _DOC_IDS + r"\s*[\]】］〕）)]|" + _DOC_IDS)
_MARKDOWN = re.compile(r"\*\*|__|`|(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)")
_TRAILING = re.compile(r"[\s。！？；，、,.!?;:：]+$")
_UNIT = re.compile(r"[㐀-鿿]|[A-Za-z]+|\d+(?:\.\d+)?")
_META = re.compile(
    r"^(?:(?:根據|依據|依照|參考)(?:本次|以上|上述|目前)?(?:提供的|所提供的)?"
    r"(?:教材|文件|資料|參考文件|說明|內容)(?:內容)?|參考(?:文件|資料)(?:[:：].*)?|"
    r"以下(?:是|為)?(?:說明|重點|整理)|結論|總結(?:來說)?|重點)$"
)


def clean(text: str) -> str:
    """claim_text: markdown + citation markers removed, trailing punctuation stripped."""
    t = _MARKER.sub("", text)
    t = _MARKDOWN.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return _TRAILING.sub("", t).strip()


def units(text: str) -> int:
    return len(_UNIT.findall(text))


def _keep(text: str) -> bool:
    return units(text) >= MIN_UNITS and not _META.match(text)


def _pieces(answer: str) -> list[tuple[int, int]]:
    """(start, end) of raw pieces; sentence-final punctuation stays with its piece."""
    out, pos = [], 0
    for m in _BOUNDARY.finditer(answer):
        end = m.end() if m.group().strip() else m.start()  # keep 。 etc., drop newlines
        if answer[pos:end].strip():
            out.append((pos, end))
        pos = m.end()
    if answer[pos:].strip():
        out.append((pos, len(answer)))
    # connective split inside each piece (not for concessive "雖然A，但B")
    split = []
    for s, e in out:
        cut = s
        body = _LEADING.match(answer, s)
        if _CONCESSIVE.match(answer, body.end() if body else s):
            split.append((s, e))
            continue
        for m in _CONNECTIVE.finditer(answer, s, e):
            split.append((cut, m.end()))
            cut = m.end()
        split.append((cut, e))
    return split


def segment_zh(answer: str) -> list[Claim]:
    if not answer or not answer.strip():
        return []
    claims: list[Claim] = []
    for s, e in _pieces(answer):
        lead = _LEADING.match(answer, s)
        if lead:
            s = lead.end()
        while e > s and answer[e - 1].isspace():
            e -= 1
        if s >= e:
            continue
        lead_mark = _MARKER.match(answer, s)
        if lead_mark and claims:
            # "…權益。【doc_016】雇主…": a marker right after the previous 。 belongs to that claim
            prev = claims[-1]
            claims[-1] = Claim(prev.claim_text, prev.char_start, lead_mark.end())
            s = lead_mark.end()
            while s < e and answer[s].isspace():
                s += 1
            if s >= e:
                continue
        raw = answer[s:e]
        if _HEADING.fullmatch(raw.strip()):
            continue  # bold heading such as "**自行保存出勤證明**" is a title, not a claim
        text = clean(raw)
        if not text and _MARKER.search(raw) and claims:
            # marker-only piece -> belongs to the previous claim
            prev = claims[-1]
            claims[-1] = Claim(prev.claim_text, prev.char_start, e)
            continue
        if _keep(text):
            claims.append(Claim(claim_text=text, char_start=s, char_end=e))
    return claims
