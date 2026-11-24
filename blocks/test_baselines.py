"""Tests for Block 5 baselines (M-RAG)."""
import glob
import json
from pathlib import Path

import pytest

from blocks.aligner import AlignedClaim, align_bundle
from blocks.baselines import PresenceOnlyScorer
from blocks.scorer import Scorer, ScorerResult

EXAMPLES = sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "examples" / "answer_bundle_0*.json")))


def _claim(alignment: str = "nearest", doc_id: str | None = "p1") -> AlignedClaim:
    text = "Roth IRA contributions are not tax-deductible"
    return AlignedClaim(
        claim_text=text, char_start=7, char_end=7 + len(text),
        cited_doc_id=doc_id if alignment != "none" else None,
        cited_span_text="You cannot deduct contributions to a Roth IRA." if alignment != "none" else None,
        alignment=alignment,
    )


DOCS = [{"doc_id": "p1", "text": "You cannot deduct contributions to a Roth IRA."}]


class TestPresenceOnlyScorer:
    @pytest.fixture(scope="class")
    def scorer(self):
        return PresenceOnlyScorer()

    def test_satisfies_scorer_protocol(self, scorer):
        assert isinstance(scorer, Scorer)

    def test_method_name(self, scorer):
        assert scorer.name == "presence-only"

    @pytest.mark.parametrize("alignment", ["explicit", "nearest"])
    def test_cited_existing_doc_scores_one(self, scorer, alignment):
        assert scorer.score(_claim(alignment), DOCS).score == 1.0

    def test_alignment_none_scores_zero(self, scorer):
        assert scorer.score(_claim("none"), DOCS).score == 0.0

    def test_cited_doc_missing_scores_zero(self, scorer):
        assert scorer.score(_claim(doc_id="p9"), DOCS).score == 0.0

    def test_empty_cited_docs_scores_zero(self, scorer):
        assert scorer.score(_claim(), []).score == 0.0

    def test_ignores_content(self, scorer):
        """The floor's known blind spot: a wrong claim citing a real doc still passes."""
        wrong = AlignedClaim(
            claim_text="Roth IRA contributions are fully tax-deductible",
            char_start=0, char_end=10, cited_doc_id="p1",
            cited_span_text=DOCS[0]["text"], alignment="nearest",
        )
        assert scorer.score(wrong, DOCS).score == 1.0

    def test_result_contract(self, scorer):
        r = scorer.score(_claim(), DOCS)
        assert isinstance(r, ScorerResult)
        assert r.claim_id == "c7"
        assert r.cost_usd == 0.0
        assert r.deterministic is True
        assert r.latency_ms >= 0.0

    def test_determinism_three_runs(self, scorer):
        scores = {scorer.score(_claim(), DOCS).score for _ in range(3)}
        assert len(scores) == 1


@pytest.mark.skipif(not EXAMPLES, reason="examples/ not found")
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: Path(p).stem)
def test_end_to_end_on_examples(path):
    """AnswerBundle -> segment+align (Emma's Block 3/4) -> presence-only (Block 5)."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    claims = align_bundle(bundle)
    assert claims, "segmenter produced no claims"
    scorer = PresenceOnlyScorer()
    for c in claims:
        r = scorer.score(c, bundle["citations"])
        assert r.score == (0.0 if c.alignment == "none" else 1.0)
