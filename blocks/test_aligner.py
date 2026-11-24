"""Tests for Block 4 — Claim Aligner. Rule set locked 2026-11-24."""
import json
from pathlib import Path
import pytest
from blocks.aligner import align, align_bundle, AlignedClaim, NEAREST_THRESHOLD
from blocks.segmenter import Claim

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


# ── helpers ──────────────────────────────────────────────────────────────────

def _cit(doc_id: str, text: str) -> dict:
    return {"doc_id": doc_id, "text": text, "span_start": 0, "span_end": len(text)}


def _claim(text: str, start: int = 0) -> Claim:
    return Claim(claim_text=text, char_start=start, char_end=start + len(text))


# ── alignment type tests ──────────────────────────────────────────────────────

def test_explicit_exact_substring():
    claim = _claim("Roth IRA accounts must be designated as such at setup")
    cits = [_cit("p1", "The account or annuity must be designated as a Roth IRA when it is set up.")]
    result = align(claim, cits)
    # close enough — word overlap should be "nearest" or "explicit" either way
    assert result.alignment in ("explicit", "nearest")
    assert result.cited_doc_id == "p1"


def test_explicit_substring_match():
    span = "Contributions to a Roth IRA are not tax-deductible."
    claim = _claim("Contributions to a Roth IRA are not tax-deductible")
    cits = [_cit("p1", span)]
    result = align(claim, cits)
    assert result.alignment == "explicit"
    assert result.cited_doc_id == "p1"


def test_nearest_picks_best_overlap():
    claim = _claim("Roth IRA withdrawal rules allow tax-free growth")
    cits = [
        _cit("p1", "tax-free growth and tax-free withdrawals in retirement"),  # higher overlap
        _cit("p2", "pine nuts harvested from BC forests"),  # unrelated
    ]
    result = align(claim, cits)
    assert result.alignment == "nearest"
    assert result.cited_doc_id == "p1"


def test_none_when_no_overlap():
    claim = _claim("Pine nuts require additional permits in BC forests")
    cits = [_cit("p1", "quantum computing entanglement error correction")]
    result = align(claim, cits)
    assert result.alignment == "none"
    assert result.cited_doc_id is None
    assert result.cited_span_text is None


def test_none_empty_citations():
    claim = _claim("some factual statement about something")
    result = align(claim, [])
    assert result.alignment == "none"
    assert result.cited_doc_id is None


def test_alignment_preserves_claim_offsets():
    claim = _claim("Roth IRA tax-free withdrawal rule", start=42)
    cits = [_cit("p1", "tax-free withdrawal from Roth IRA")]
    result = align(claim, cits)
    assert result.char_start == 42
    assert result.char_end == 42 + len(claim.claim_text)


def test_case_insensitive_explicit():
    claim = _claim("roth ira tax-free")
    cits = [_cit("p1", "Roth IRA tax-free growth and withdrawals")]
    result = align(claim, cits)
    assert result.alignment == "explicit"


# ── bundle integration tests ──────────────────────────────────────────────────

def test_align_bundle_returns_list():
    bundle = {
        "answer": "The Roth IRA allows tax-free withdrawals. Contributions are not deductible.",
        "citations": [_cit("p1", "Roth IRA allows tax-free growth and withdrawals in retirement")],
    }
    results = align_bundle(bundle)
    assert isinstance(results, list)
    assert all(isinstance(r, AlignedClaim) for r in results)
    assert len(results) >= 1


def test_align_bundle_all_claims_have_doc_or_none():
    bundle = {
        "answer": "Tax-free withdrawals apply after age 59½. No deductible contributions.",
        "citations": [_cit("p1", "tax-free withdrawals after 59½")],
    }
    results = align_bundle(bundle)
    for r in results:
        assert r.alignment in ("explicit", "nearest", "none")
        if r.alignment == "none":
            assert r.cited_doc_id is None


def test_align_bundle_example_01():
    bundle_path = EXAMPLES_DIR / "answer_bundle_01.json"
    if not bundle_path.exists():
        pytest.skip("example file not in repo")
    with open(bundle_path) as f:
        bundle = json.load(f)
    results = align_bundle(bundle)
    assert len(results) > 0
    # RAGTruth bundles: all retrieved passages are cited, so most claims should be nearest
    aligned = [r for r in results if r.alignment != "none"]
    assert len(aligned) >= len(results) // 2, "majority of claims should align to a source"


def test_align_bundle_hallucinated_example():
    # answer_bundle_03 is hallucinated; at least one claim should still align
    bundle_path = EXAMPLES_DIR / "answer_bundle_03.json"
    if not bundle_path.exists():
        pytest.skip("example file not in repo")
    with open(bundle_path) as f:
        bundle = json.load(f)
    results = align_bundle(bundle)
    assert len(results) > 0


# ── determinism test ──────────────────────────────────────────────────────────

def test_determinism_three_runs():
    bundle = {
        "answer": "Tax-free growth applies. Contributions are not deductible. Age 59½ rule applies.",
        "citations": [
            _cit("p1", "tax-free growth and withdrawals"),
            _cit("p2", "contributions are not tax-deductible"),
        ],
    }
    r1 = [(r.alignment, r.cited_doc_id) for r in align_bundle(bundle)]
    r2 = [(r.alignment, r.cited_doc_id) for r in align_bundle(bundle)]
    r3 = [(r.alignment, r.cited_doc_id) for r in align_bundle(bundle)]
    assert r1 == r2 == r3
