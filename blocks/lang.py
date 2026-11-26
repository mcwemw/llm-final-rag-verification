"""Language-aware processing strategy for segmenter/aligner (Jimmy #7, --lang zh).

A PROCESSING strategy only — it does NOT touch the AlignedClaim / AnswerBundle contract.
Three strategies: en (default, unchanged behaviour), zh, auto (mixed, language-aware).

Design notes (Emma's constraints):
  1. We do NOT just swap word-count for char-count. Length is an abstract per-language rule:
     en = word count; zh/auto = ascii-tokens + CJK content measured in ~2-char units.
  2. Chinese alignment uses LEXICAL char-bigram overlap — it is NOT semantic entailment.
     See ZH_ALIGNMENT_METHOD; the known limitation is documented, not hidden.
  3. English abbreviations / numbers / proper nouns (COVID-19, 59, F1) are kept as whole
     tokens in mixed text — never split char-by-char.
"""
from __future__ import annotations
import re

# CJK unified ideographs (+ extension A). Enough for Traditional/Simplified Chinese.
_CJK = re.compile(r"[㐀-䶿一-鿿]")
# Whole ascii tokens to preserve in mixed text: words, numbers, COVID-19, U.S., F1, 3.14 …
_ASCII_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.\-'][A-Za-z0-9]+)*")
# English sentence/clause boundary (identical to the original segmenter rule).
_EN_BOUNDARY = r"(?<=[.!?])\s+|[\n\r]+|;\s*"
# Full-width Chinese sentence/clause enders: 。！？ (sentence) ；(clause). Split right after.
_ZH_BOUNDARY = r"(?<=[。！？；])"

_EN_WORD = re.compile(r"\b\w+\b")

MIN_WORDS_EN = 4          # unchanged English rule
MIN_UNITS_ZH = 3          # zh/auto: ascii tokens + ceil(cjk/2) >= this
ZH_ALIGNMENT_METHOD = "lexical char-bigram Jaccard (NOT semantic entailment)"


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def _cjk_chars(text: str) -> int:
    return len(_CJK.findall(text or ""))


def _ascii_tokens(text: str) -> list[str]:
    return _ASCII_TOKEN.findall(text or "")


def _char_bigrams(cjk_run: str) -> list[str]:
    if len(cjk_run) == 1:
        return [cjk_run]
    return [cjk_run[i:i + 2] for i in range(len(cjk_run) - 1)]


class LangStrategy:
    name = "en"
    boundary = re.compile(_EN_BOUNDARY, re.MULTILINE)

    def long_enough(self, text: str) -> bool:
        return len(text.split()) >= MIN_WORDS_EN

    def tokens(self, text: str) -> set[str]:
        """Token set for aligner Jaccard (English: whitespace/word tokens)."""
        return {w.lower() for w in _EN_WORD.findall(text)}


class _ZhStrategy(LangStrategy):
    name = "zh"
    # include Chinese enders AND English boundaries so mixed text still splits on either.
    boundary = re.compile(_EN_BOUNDARY + "|" + _ZH_BOUNDARY, re.MULTILINE)

    def long_enough(self, text: str) -> bool:
        units = len(_ascii_tokens(text)) + (_cjk_chars(text) + 1) // 2
        return units >= MIN_UNITS_ZH

    def tokens(self, text: str) -> set[str]:
        """ascii tokens (whole, lowercased) + CJK character bigrams (lexical overlap)."""
        toks = {t.lower() for t in _ascii_tokens(text)}
        for run in re.findall(r"[㐀-䶿一-鿿]+", text):
            toks.update(_char_bigrams(run))
        return toks


# auto == zh strategy (language-aware; same rules, applies to mixed en/zh)
EN = LangStrategy()
ZH = _ZhStrategy()
AUTO = _ZhStrategy()
AUTO.name = "auto"

_BY_NAME = {"en": EN, "zh": ZH, "auto": AUTO}


def get_strategy(lang: str) -> LangStrategy:
    s = _BY_NAME.get((lang or "en").lower())
    if s is None:
        raise ValueError(f"unknown --lang {lang!r}; use en|zh|auto")
    return s
