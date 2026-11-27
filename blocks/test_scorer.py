"""Tests for Block 5 — Grounding Scorer. Rule set locked 2026-11-24."""
import pytest
from blocks.aligner import AlignedClaim
from blocks.scorer import ScorerResult, SelfAttentionScorer, Scorer


# ── helpers ──────────────────────────────────────────────────────────────────

def _claim(text: str, alignment: str = "nearest", cited_text: str = "") -> AlignedClaim:
    return AlignedClaim(
        claim_text=text,
        char_start=0,
        char_end=len(text),
        cited_doc_id="p1" if alignment != "none" else None,
        cited_span_text=cited_text or (text if alignment != "none" else None),
        alignment=alignment,
    )


def _docs(text: str) -> list[dict]:
    return [{"doc_id": "p1", "text": text}]


# ── ScorerResult contract ─────────────────────────────────────────────────────

def test_scorer_result_fields():
    r = ScorerResult(claim_id="c0", method="hw1-selfattn", score=0.5,
                     latency_ms=1.0, cost_usd=0.0, deterministic=True)
    assert 0.0 <= r.score <= 1.0
    assert r.latency_ms >= 0.0
    assert r.cost_usd == 0.0
    assert r.deterministic is True


def test_scorer_result_score_bounds():
    for s in (0.0, 0.5, 1.0):
        r = ScorerResult(claim_id="c0", method="hw1-selfattn", score=s,
                         latency_ms=0.0, cost_usd=0.0)
        assert 0.0 <= r.score <= 1.0


# ── NLIScorer label mapping ───────────────────────────────────────────────────

def test_nli_label_index_by_name():
    # cross-encoder/nli-deberta-v3-small: {0: contradiction, 1: entailment, 2: neutral};
    # the old code read index 2 (= neutral) as entailment.
    import pytest
    from blocks.scorer import _label_index
    assert _label_index({0: "contradiction", 1: "entailment", 2: "neutral"}, "entailment") == 1
    assert _label_index({"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}, "entailment") == 2
    with pytest.raises(ValueError):
        _label_index({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, "entailment")


def test_nli_scorer_entailment_is_high_for_entailed_claim():
    import pytest
    pytest.importorskip("sentence_transformers")
    from blocks.aligner import AlignedClaim
    from blocks.scorer import NLIScorer
    s = NLIScorer()
    def claim(text):
        return AlignedClaim(claim_text=text, char_start=0, char_end=len(text), cited_doc_id="d1",
                            cited_span_text="The cat sat on the mat.", alignment="explicit")
    docs = [{"doc_id": "d1", "text": "The cat sat on the mat."}]
    entailed = s.score(claim("A cat is on the mat."), docs).score
    unrelated = s.score(claim("The cat is black."), docs).score   # neutral, not entailed
    assert entailed > 0.9 and unrelated < 0.1


# ── SelfAttentionScorer — full tests (no external deps) ───────────────────────

class TestSelfAttentionScorer:
    @pytest.fixture(scope="class")
    def scorer(self):
        return SelfAttentionScorer()

    def test_method_name(self, scorer):
        assert scorer.name == "hw1-selfattn"

    def test_score_range(self, scorer):
        claim = _claim("Roth IRA tax-free withdrawal rule", "nearest",
                       "tax-free withdrawals from Roth IRA accounts")
        result = scorer.score(claim, _docs("tax-free withdrawals from Roth IRA accounts"))
        assert 0.0 <= result.score <= 1.0

    def test_exact_match_high_score(self, scorer):
        text = "Roth IRA allows tax-free withdrawals after age fifty nine"
        claim = _claim(text, "explicit", text)
        result = scorer.score(claim, _docs(text))
        assert result.score > 0.5, "identical text should score > 0.5"

    def test_unrelated_low_score(self, scorer):
        claim = _claim("quantum computing uses qubit superposition states",
                       "nearest", "pine nuts harvested from BC forests")
        result = scorer.score(claim, _docs("pine nuts harvested from BC forests"))
        related = scorer.score(
            _claim("pine nuts are expensive forest products", "nearest",
                   "pine nuts harvested from BC forests"),
            _docs("pine nuts harvested from BC forests")
        )
        assert result.score < related.score

    def test_alignment_none_returns_zero(self, scorer):
        claim = _claim("some claim text here now", "none")
        result = scorer.score(claim, _docs("some claim text here now"))
        assert result.score == 0.0

    def test_empty_cited_docs_returns_zero(self, scorer):
        claim = _claim("Roth IRA tax-free withdrawal rule", "nearest",
                       "tax-free withdrawals from Roth IRA")
        result = scorer.score(claim, [])
        assert result.score == 0.0

    def test_determinism_three_runs(self, scorer):
        claim = _claim("Roth IRA allows tax-free growth and withdrawals retirement",
                       "nearest", "tax-free growth and tax-free withdrawals in retirement")
        docs = _docs("tax-free growth and tax-free withdrawals in retirement")
        r1 = scorer.score(claim, docs)
        r2 = scorer.score(claim, docs)
        r3 = scorer.score(claim, docs)
        assert r1.score == r2.score == r3.score, "SelfAttentionScorer must be deterministic"

    def test_claim_id_matches_char_start(self, scorer):
        claim = AlignedClaim(
            claim_text="test claim for id check",
            char_start=42,
            char_end=65,
            cited_doc_id="p1",
            cited_span_text="test claim for id check",
            alignment="nearest",
        )
        result = scorer.score(claim, _docs("test claim for id check"))
        assert result.claim_id == "c42"

    def test_deterministic_flag_true(self, scorer):
        claim = _claim("some factual statement about something", "nearest",
                       "some factual statement about something indeed")
        result = scorer.score(claim, _docs("some factual statement about something indeed"))
        assert result.deterministic is True

    def test_cost_usd_zero(self, scorer):
        claim = _claim("Roth IRA tax-free withdrawal", "nearest",
                       "Roth IRA allows tax-free withdrawals after fifty nine")
        result = scorer.score(claim, _docs("Roth IRA allows tax-free withdrawals after fifty nine"))
        assert result.cost_usd == 0.0

    def test_latency_ms_non_negative(self, scorer):
        claim = _claim("short claim about Roth IRA rules here", "nearest",
                       "Roth IRA rules allow tax-free growth accounts")
        result = scorer.score(claim, _docs("Roth IRA rules allow tax-free growth accounts"))
        assert result.latency_ms >= 0.0

    def test_empty_claim_text_no_crash(self, scorer):
        claim = AlignedClaim("", 0, 0, "p1", "some passage text here please", "nearest")
        result = scorer.score(claim, _docs("some passage text here please"))
        assert 0.0 <= result.score <= 1.0


# ── EmbeddingScorer — skipped if sentence_transformers not installed ───────────

class TestEmbeddingScorer:
    @pytest.fixture(scope="class")
    def scorer(self):
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence_transformers not installed; skip embedding tests",
        )
        from blocks.scorer import EmbeddingScorer
        return EmbeddingScorer()

    def test_method_name(self, scorer):
        assert scorer.name == "bge-m3-cosine"

    def test_score_range(self, scorer):
        claim = _claim("Roth IRA tax-free withdrawal rule", "nearest",
                       "tax-free withdrawals from Roth IRA accounts")
        result = scorer.score(claim, _docs("tax-free withdrawals from Roth IRA accounts"))
        assert 0.0 <= result.score <= 1.0

    def test_determinism_three_runs(self, scorer):
        claim = _claim("Precision measures exactness of relevant retrieved results",
                       "nearest", "Precision is the fraction of retrieved documents that are relevant")
        docs = _docs("Precision is the fraction of retrieved documents that are relevant")
        r1 = scorer.score(claim, docs).score
        r2 = scorer.score(claim, docs).score
        r3 = scorer.score(claim, docs).score
        assert r1 == r2 == r3, "EmbeddingScorer must be deterministic (3-run variance = 0)"

    def test_alignment_none_returns_zero(self, scorer):
        claim = _claim("some claim text here now", "none")
        result = scorer.score(claim, _docs("some claim text here"))
        assert result.score == 0.0

    def test_deterministic_flag_true(self, scorer):
        claim = _claim("Roth IRA accounts tax free withdrawal", "nearest",
                       "Roth IRA allows tax-free withdrawals")
        result = scorer.score(claim, _docs("Roth IRA allows tax-free withdrawals"))
        assert result.deterministic is True
