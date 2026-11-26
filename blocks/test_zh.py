"""Tests for the Chinese segmenter / aligner proposal (--lang zh)."""
import pytest

from blocks.aligner_zh import align_bundle_zh, grams, jaccard, markers
from blocks.segmenter_zh import clean, segment_zh, units

DOCS = [
    {"doc_id": "doc_016", "span_start": 0, "span_end": 40,
     "text": "工讀生、暑期短期工作者及兼職者依法享有勞動權益，雇主不得因為是學生就任意降低工資。"},
    {"doc_id": "doc_004", "span_start": 0, "span_end": 30,
     "text": "正常工時原則為每日不超過八小時、每週不超過四十小時，超過者為延長工時。"},
]


def _texts(answer):
    return [c.claim_text for c in segment_zh(answer)]


class TestSegmenterZh:
    def test_splits_on_chinese_sentence_punctuation(self):
        a = "工讀生依法享有勞動權益。雇主不得任意降低工資！學生也受工時保護？"
        assert _texts(a) == ["工讀生依法享有勞動權益", "雇主不得任意降低工資", "學生也受工時保護"]

    def test_semicolon_and_newline(self):
        assert _texts("每日正常工時不超過八小時；每週不超過四十小時\n超過部分屬於延長工時") == [
            "每日正常工時不超過八小時", "每週不超過四十小時", "超過部分屬於延長工時"]

    def test_decimal_point_is_not_a_boundary(self):
        assert _texts("加班費率為 1.33 倍與 1.66 倍。") == ["加班費率為 1.33 倍與 1.66 倍"]

    def test_markdown_list_and_bold_removed(self):
        a = "1. **基本權益**  \n   工讀生**依法**享有勞動權益。\n   - 雇主不得任意降低工資。"
        assert _texts(a) == ["工讀生依法享有勞動權益", "雇主不得任意降低工資"]

    def test_bold_heading_line_skipped(self):
        assert _texts("**自行保存出勤證明**\n請保存打卡照片與排班表。") == ["請保存打卡照片與排班表"]

    @pytest.mark.parametrize("marker", ["【doc_016】", "[doc_016]", "〔doc_016〕", "（doc_016）", "［doc_016］",
                                        "[doc_016, doc_004]", "【doc_016、doc_004】"])
    def test_citation_markers_removed_from_text_but_kept_in_span(self, marker):
        a = f"工讀生依法享有勞動權益{marker}。"
        (c,) = segment_zh(a)
        assert c.claim_text == "工讀生依法享有勞動權益"
        assert "doc_016" in a[c.char_start:c.char_end]

    def test_marker_after_period_merges_into_previous_claim(self):
        a = "工讀生依法享有勞動權益。【doc_016】雇主不得任意降低工資。"
        cs = segment_zh(a)
        assert [c.claim_text for c in cs] == ["工讀生依法享有勞動權益", "雇主不得任意降低工資"]
        assert "doc_016" in a[cs[0].char_start:cs[0].char_end]

    def test_contrast_connective_split(self):
        assert _texts("Voyage 有 3G 版本，但 Paperwhite 沒有 3G 版本。") == [
            "Voyage 有 3G 版本", "但 Paperwhite 沒有 3G 版本"]

    def test_concessive_not_split(self):
        assert _texts("雖然是兼職身分，但仍然可以依法請假。") == ["雖然是兼職身分，但仍然可以依法請假"]

    def test_er_qie_not_split(self):
        assert _texts("雇主必須保存出勤紀錄，而且要保存五年。") == ["雇主必須保存出勤紀錄，而且要保存五年"]

    @pytest.mark.parametrize("meta", ["根據提供的文件：", "參考文件：請假權益（doc_011）", "結論：", "以下是說明："])
    def test_meta_only_pieces_dropped(self, meta):
        assert _texts(meta) == []

    def test_min_units(self):
        assert units("勞基法") == 3 and _texts("勞基法。") == []
        assert units("每週 40 小時") == 5
        assert units("工讀生有特休嗎") == 7

    def test_offsets_cover_raw_text(self):
        a = "是的。\n\n1. **工讀生**依法享有勞動權益【doc_016】。"
        for c in segment_zh(a):
            assert clean(a[c.char_start:c.char_end]) == c.claim_text

    def test_empty(self):
        assert segment_zh("") == [] and segment_zh("  \n ") == []


class TestAlignerZh:
    def test_marker_gives_explicit(self):
        b = {"answer": "雇主不得因為是學生就任意降低工資【doc_016】。", "citations": DOCS}
        (c,) = align_bundle_zh(b)
        assert c.alignment == "explicit" and c.cited_doc_id == "doc_016"
        assert c.cited_span_text == DOCS[0]["text"]

    def test_multiple_markers_pick_best_overlap(self):
        b = {"answer": "每日正常工時不超過八小時【doc_016、doc_004】。", "citations": DOCS}
        assert align_bundle_zh(b)[0].cited_doc_id == "doc_004"

    def test_no_marker_falls_back_to_bigram_nearest(self):
        b = {"answer": "每週正常工時不超過四十小時。", "citations": DOCS}
        (c,) = align_bundle_zh(b)
        assert c.alignment == "nearest" and c.cited_doc_id == "doc_004"

    def test_unrelated_claim_is_none(self):
        b = {"answer": "政大附近的咖啡店環境很舒適。", "citations": DOCS}
        (c,) = align_bundle_zh(b)
        assert c.alignment == "none" and c.cited_doc_id is None

    def test_marker_to_uncited_doc_is_not_explicit(self):
        b = {"answer": "政大附近的咖啡店環境很舒適【doc_099】。", "citations": DOCS}
        assert align_bundle_zh(b)[0].alignment == "none"

    def test_no_citations_is_none(self):
        assert align_bundle_zh({"answer": "工讀生依法享有勞動權益。", "citations": []})[0].alignment == "none"

    def test_english_word_jaccard_fails_but_bigrams_work(self):
        """Why a zh path is needed: \\w+ sees a whole Chinese sentence as one word."""
        from blocks.aligner import _jaccard, _words
        claim, doc = "雇主不得任意降低工資", DOCS[0]["text"]
        assert _jaccard(_words(claim), _words(doc)) == 0.0
        assert jaccard(grams(claim), grams(doc)) > 0.05

    def test_markers_helper(self):
        assert markers("見【doc_016】與 [doc_003, doc_010]") == ["doc_016", "doc_003", "doc_010"]
