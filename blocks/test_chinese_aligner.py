"""Chinese / mixed aligner tests (Jimmy #7, --lang zh). SEPARATE from English tests.
Chinese alignment = LEXICAL char-bigram overlap (NOT semantic) — these tests assert the
real behaviour AND pin the known limitation so it is surfaced, not faked."""
from blocks.aligner import align, align_bundle, AlignedClaim
from blocks.segmenter import Claim
from blocks.lang import ZH_ALIGNMENT_METHOD


def _cit(doc_id, text):
    return {"doc_id": doc_id, "text": text, "span_start": 0, "span_end": len(text)}


def _claim(text, start=0):
    return Claim(claim_text=text, char_start=start, char_end=start + len(text))


# ── Chinese aligns (no longer always 'none') ──────────────────────────────────

def test_chinese_explicit_substring():
    claim = _claim("台北是台灣的首都")
    cits = [_cit("p1", "根據資料，台北是台灣的首都，位於北部。")]
    r = align(claim, cits, lang="zh")
    assert r.alignment == "explicit"
    assert r.cited_doc_id == "p1"


def test_chinese_nearest_via_bigram_overlap():
    claim = _claim("台灣的首都位於台北")
    cits = [
        _cit("p1", "台灣的首都是台北市，人口眾多"),   # high char-bigram overlap
        _cit("p2", "量子電腦使用疊加與糾纏"),          # unrelated
    ]
    r = align(claim, cits, lang="zh")
    assert r.alignment == "nearest", "lexically-overlapping Chinese must NOT be 'none'"
    assert r.cited_doc_id == "p1"


def test_chinese_none_when_no_char_overlap():
    claim = _claim("越野車適合山區地形行駛")
    cits = [_cit("p1", "量子電腦錯誤更正與疊加糾纏")]
    r = align(claim, cits, lang="zh")
    assert r.alignment == "none"
    assert r.cited_doc_id is None


def test_known_limitation_paraphrase_is_none_not_faked():
    # paraphrase with NO shared characters -> lexical method returns 'none'.
    # This is the DOCUMENTED limitation: we do not claim semantic zh alignment.
    claim = _claim("汽車是陸上運輸的主要工具")
    cits = [_cit("p1", "車輛為地面載運乘客的設備")]
    r = align(claim, cits, lang="zh")
    assert r.alignment == "none"
    assert "NOT semantic" in ZH_ALIGNMENT_METHOD   # limitation surfaced, not hidden


def test_mixed_claim_with_english_term():
    claim = _claim("RAG系統使用檢索增強生成")
    cits = [_cit("p1", "RAG系統透過檢索增強生成來回答問題")]
    r = align(claim, cits, lang="zh")
    assert r.alignment in ("explicit", "nearest")
    assert r.cited_doc_id == "p1"


# ── bundle + contract unchanged ───────────────────────────────────────────────

def test_chinese_bundle_contract_unchanged():
    bundle = {
        "answer": "台北是台灣的首都。愛因斯坦提出了相對論。",
        "citations": [_cit("p1", "台北是台灣的首都，位於北部地區")],
    }
    results = align_bundle(bundle, lang="zh")
    assert all(isinstance(r, AlignedClaim) for r in results)
    assert all(r.alignment in ("explicit", "nearest", "none") for r in results)
    # AlignedClaim must have exactly the original fields (no contract change)
    assert set(AlignedClaim.__dataclass_fields__) == {
        "claim_text", "char_start", "char_end", "cited_doc_id", "cited_span_text", "alignment"}


def test_chinese_determinism():
    bundle = {"answer": "台北是台灣的首都。量子計算使用疊加糾纏加速。",
              "citations": [_cit("p1", "台北是台灣的首都"), _cit("p2", "量子計算使用疊加與糾纏")]}
    r1 = [(r.alignment, r.cited_doc_id) for r in align_bundle(bundle, lang="zh")]
    r2 = [(r.alignment, r.cited_doc_id) for r in align_bundle(bundle, lang="zh")]
    assert r1 == r2
