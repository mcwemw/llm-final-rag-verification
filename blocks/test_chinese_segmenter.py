"""Chinese / mixed segmenter tests (Jimmy #7, --lang zh). Kept SEPARATE from the English
main-experiment tests. Covers Emma's acceptance table for segmentation."""
from blocks.segmenter import segment


# ── Chinese sentence boundaries ───────────────────────────────────────────────

def test_chinese_period_splits():
    answer = "台灣的首都是台北。愛因斯坦發明了相對論。"
    claims = segment(answer, lang="zh")
    assert len(claims) == 2
    assert claims[0].claim_text == "台灣的首都是台北"
    assert "愛因斯坦" in claims[1].claim_text


def test_chinese_question_exclamation_split():
    answer = "這是真的嗎？這個敘述也完全正確！"
    claims = segment(answer, lang="zh")
    assert len(claims) == 2


def test_chinese_semicolon_splits_clause():
    answer = "越野車適合off-road使用；多用途車輛適合農業運輸作業。"
    claims = segment(answer, lang="zh")
    assert len(claims) == 2


# ── pure Chinese no longer vanishes (the core bug) ────────────────────────────

def test_pure_chinese_not_dropped():
    answer = "量子計算使用疊加與糾纏來加速運算處理。"
    en = segment(answer, lang="en")     # English rule: 0 whitespace-words -> dropped
    zh = segment(answer, lang="zh")
    assert len(en) == 0, "English rule drops pure Chinese (documents the original bug)"
    assert len(zh) == 1, "zh must keep the Chinese claim"


def test_very_short_chinese_filtered():
    # a 2-char fragment is below the content threshold -> dropped (not every char is a word)
    assert segment("好的。", lang="zh") == []


# ── mixed: preserve English abbreviations / numbers ───────────────────────────

def test_mixed_preserves_abbreviation_and_number():
    answer = "COVID-19 在2021年造成全球大流行。"
    claims = segment(answer, lang="zh")
    assert len(claims) == 1
    assert "COVID-19" in claims[0].claim_text   # abbreviation not split char-by-char
    assert "2021" in claims[0].claim_text


def test_mixed_english_sentence_under_zh_still_splits():
    answer = "F1 balances precision and recall。這是中文句子的測試。"
    claims = segment(answer, lang="zh")
    assert len(claims) == 2


# ── edges ─────────────────────────────────────────────────────────────────────

def test_empty_and_punct_only():
    assert segment("", lang="zh") == []
    assert segment("   ", lang="zh") == []
    assert segment("。！？；", lang="zh") == []


def test_chinese_offsets_valid():
    answer = "第一個句子描述了重要的事實。第二個句子補充更多細節內容。"
    for c in segment(answer, lang="zh"):
        assert 0 <= c.char_start < c.char_end <= len(answer)
        assert answer[c.char_start:c.char_end].strip().startswith(c.claim_text[:2])


def test_auto_lang_behaves_like_zh_for_mixed():
    answer = "台灣首都是台北。Precision measures exactness of items."
    claims = segment(answer, lang="auto")
    assert len(claims) == 2
