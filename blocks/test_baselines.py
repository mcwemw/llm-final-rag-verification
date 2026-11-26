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


# ── RagasScorer — offline tests (fake Groq server, no API key / network) ──────

import httpx  # noqa: E402

from blocks.baselines import RagasScorer  # noqa: E402


def _fake_groq_client(verdicts, prompt_tokens=100, completion_tokens=20):
    """AsyncOpenAI client whose HTTP layer is a local fake of Groq's chat endpoint.

    Call 1 (statement generation) -> {"statements": [...]}
    Call 2 (NLI verdicts)         -> {"statements": [{statement, reason, verdict}, ...]}
    """
    from openai import AsyncOpenAI
    calls = {"n": 0, "bodies": []}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        calls["bodies"].append(json.loads(request.content))
        stmts = [f"s{i}" for i in range(len(verdicts))]
        if calls["n"] % 2 == 1:
            content = {"statements": stmts}
        else:
            content = {"statements": [{"statement": s, "reason": "r", "verdict": v}
                                      for s, v in zip(stmts, verdicts)]}
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(content)}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        })

    client = AsyncOpenAI(api_key="test", base_url="https://fake.groq/openai/v1",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return client, calls


@pytest.fixture
def ragas_factory():
    made = []

    def make(verdicts, **kw):
        client, calls = _fake_groq_client(verdicts)
        s = RagasScorer(client=client, price_in_per_mtok=kw.pop("pin", 1.0),
                        price_out_per_mtok=kw.pop("pout", 2.0), **kw)
        made.append(s)
        return s, calls

    yield make
    for s in made:
        s.close()


class TestRagasScorer:
    def test_satisfies_scorer_protocol(self, ragas_factory):
        s, _ = ragas_factory([1])
        assert isinstance(s, Scorer)
        assert s.name == "ragas-llm-judge"

    @pytest.mark.parametrize("verdicts,expected", [([1, 1], 1.0), ([1, 0], 0.5), ([0, 0, 0], 0.0)])
    def test_score_is_supported_ratio(self, ragas_factory, verdicts, expected):
        s, calls = ragas_factory(verdicts)
        s.set_question("Can contributions be deducted?")
        r = s.score(_claim(), DOCS)
        assert r.score == pytest.approx(expected)
        assert calls["n"] == 2                      # statement generation + NLI verdicts
        assert r.deterministic is False

    def test_cost_from_token_usage(self, ragas_factory):
        s, _ = ragas_factory([1], pin=1.0, pout=2.0)   # USD per 1M tokens
        r = s.score(_claim(), DOCS)
        # 2 calls x (100 in, 20 out) -> 200*1 + 40*2 = 280 / 1e6
        assert r.cost_usd == pytest.approx(280 / 1e6)
        assert s.total_usage == {"prompt_tokens": 200, "completion_tokens": 40, "calls": 2}

    def test_sends_temperature_zero_and_model(self, ragas_factory):
        s, calls = ragas_factory([1])
        s.score(_claim(), DOCS)
        body = calls["bodies"][0]
        assert body["temperature"] == 0.0
        assert body["model"] == s.model

    def test_question_fallback_counted(self, ragas_factory):
        s, _ = ragas_factory([1])
        s.score(_claim(), DOCS)
        assert s.n_fallback_question == 1
        s.set_question("q?")
        s.score(_claim(), DOCS)
        assert s.n_fallback_question == 1

    def test_alignment_none_skips_llm(self, ragas_factory):
        s, calls = ragas_factory([1])
        r = s.score(_claim("none"), DOCS)
        assert r.score == 0.0 and r.cost_usd == 0.0 and calls["n"] == 0

    def test_empty_statements_nan_counted_as_zero(self, ragas_factory):
        s, _ = ragas_factory([])                   # judge returns no statements -> NaN
        r = s.score(_claim(), DOCS)
        assert r.score == 0.0 and s.n_nan == 1

    def test_many_calls_reuse_one_event_loop(self, ragas_factory):
        s, calls = ragas_factory([1])
        for _ in range(3):
            assert s.score(_claim(), DOCS).score == 1.0
        assert calls["n"] == 6

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            RagasScorer(price_in_per_mtok=0, price_out_per_mtok=0)
