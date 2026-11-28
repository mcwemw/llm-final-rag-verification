"""Tests for Block 5 baselines (M-RAG)."""
import glob
import math
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
        assert {k: s.total_usage[k] for k in ("prompt_tokens", "completion_tokens", "calls")} == \
            {"prompt_tokens": 200, "completion_tokens": 40, "calls": 2}

    def test_sends_temperature_zero_top_p_one_and_model(self, ragas_factory):
        s, calls = ragas_factory([1])
        s.score(_claim(), DOCS)
        for body in calls["bodies"]:
            assert body["temperature"] == 0.0
            assert body["top_p"] == 1.0
            assert body["model"] == s.model
            assert body.get("max_tokens", body.get("max_completion_tokens")) == 4096

    def test_status(self, ragas_factory):
        s, _ = ragas_factory([1])
        s.score(_claim(), DOCS)
        assert s.last_status == "ok"
        s.score(_claim("none"), DOCS)
        assert s.last_status == "skipped_no_source"

    def test_per_claim_usage_resets(self, ragas_factory):
        s, _ = ragas_factory([1])
        s.score(_claim(), DOCS)
        s.score(_claim(), DOCS)
        assert s.last_usage["calls"] == 2 and s.total_usage["calls"] == 4

    def test_config_records_provider_and_prompt_version(self, ragas_factory):
        s, _ = ragas_factory([1])
        c = s.config
        assert c["provider"] == "groq" and c["top_p"] == 1.0 and c["cache"] is None
        assert c["prompt_version"].startswith("ragas==")

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

    def test_empty_statements_is_invalid_output_not_zero(self, ragas_factory):
        s, _ = ragas_factory([])                   # judge returns no statements -> NaN
        r = s.score(_claim(), DOCS)
        assert math.isnan(r.score) and r.score != 0.0
        assert s.last_status == "invalid_output" and s.n_nan == 1 and s.last_invalid_reason
        assert s.score(_claim(), DOCS) is not None and s.n_nan == 2   # counted every time

    def test_many_calls_reuse_one_event_loop(self, ragas_factory):
        s, calls = ragas_factory([1])
        for _ in range(3):
            assert s.score(_claim(), DOCS).score == 1.0
        assert calls["n"] == 6

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            RagasScorer(price_in_per_mtok=0, price_out_per_mtok=0)


# ── SelfCheckGPTScorer — offline tests (fake Groq + fake NLI, no key / GPU) ───

from blocks.baselines import SelfCheckGPTScorer  # noqa: E402


def _fake_sync_groq(samples, prompt_tokens=300, completion_tokens=50):
    """Sync OpenAI client returning `samples` in turn (cycled) as chat completions."""
    from openai import OpenAI
    calls = {"n": 0, "bodies": []}

    def handler(request: httpx.Request) -> httpx.Response:
        text = samples[calls["n"] % len(samples)]
        calls["n"] += 1
        calls["bodies"].append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        })

    client = OpenAI(api_key="test", base_url="https://fake.groq/openai/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, calls


def _fake_nli(contra_by_sample):
    """P(contradiction) looked up by sample text; records the pairs it was given."""
    seen = []

    def nli(pairs):
        seen.extend(pairs)
        return [contra_by_sample[s] for _, s in pairs]

    return nli, seen


@pytest.fixture
def sc_factory():
    def make(samples, contra, **kw):
        client, calls = _fake_sync_groq(samples)
        nli, seen = _fake_nli(contra)
        s = SelfCheckGPTScorer(client=client, nli=nli, price_in_per_mtok=kw.pop("pin", 1.0),
                               price_out_per_mtok=kw.pop("pout", 2.0), **kw)
        return s, calls, seen
    return make


class TestSelfCheckGPTScorer:
    def test_satisfies_scorer_protocol(self, sc_factory):
        s, _, _ = sc_factory(["a"], {"a": 0.0})
        assert isinstance(s, Scorer)
        assert s.name == "selfcheckgpt-nli"

    def test_score_is_one_minus_mean_contradiction(self, sc_factory):
        s, calls, seen = sc_factory(["a", "b", "c"], {"a": 0.0, "b": 0.3, "c": 0.9}, n_samples=3)
        s.set_question("Can contributions be deducted?")
        r = s.score(_claim(), DOCS)
        assert r.score == pytest.approx(1 - (0.0 + 0.3 + 0.9) / 3)
        assert calls["n"] == 3
        assert r.deterministic is False
        # official pair order: (sentence, sample)
        assert seen[0] == (_claim().claim_text, "a")

    def test_samples_shared_within_response(self, sc_factory):
        s, calls, _ = sc_factory(["a"], {"a": 0.1}, n_samples=3)
        s.set_question("q?")
        r1 = s.score(_claim(), DOCS)
        r2 = s.score(_claim(), DOCS)            # second claim of the same response
        assert calls["n"] == 3                  # sampled once
        assert r1.cost_usd == pytest.approx((900 * 1 + 150 * 2) / 1e6)
        assert r2.cost_usd == 0.0
        assert {k: s.total_usage[k] for k in ("prompt_tokens", "completion_tokens", "calls")} == \
            {"prompt_tokens": 900, "completion_tokens": 150, "calls": 3}

    def test_reset_samples_resamples(self, sc_factory):
        s, calls, _ = sc_factory(["a"], {"a": 0.1}, n_samples=2)
        s.set_question("q?")
        s.score(_claim(), DOCS)
        s.reset_samples()
        s.score(_claim(), DOCS)
        assert calls["n"] == 4

    def test_new_question_resamples(self, sc_factory):
        s, calls, _ = sc_factory(["a"], {"a": 0.1}, n_samples=2)
        s.set_question("q1?"); s.score(_claim(), DOCS)
        s.set_question("q2?"); s.score(_claim(), DOCS)
        assert calls["n"] == 4

    def test_request_uses_ragtruth_prompt_and_settings(self, sc_factory):
        s, calls, _ = sc_factory(["a"], {"a": 0.0}, n_samples=1)
        s.set_question("Can contributions be deducted?")
        s.score(_claim(), DOCS)
        body = calls["bodies"][0]
        assert body["temperature"] == 1.0
        assert body["model"] == s.model
        assert body["reasoning_effort"] == "low"
        assert body["top_p"] == 1.0
        prompt = body["messages"][0]["content"]
        assert prompt.startswith("Briefly answer the following question:\nCan contributions be deducted?\n")
        assert "following one passages:\npassage 1:You cannot deduct" in prompt
        assert prompt.endswith('"Unable to answer based on given passages."\noutput:')

    def test_empty_samples_dropped_and_counted(self, sc_factory):
        s, _, seen = sc_factory(["", "a"], {"a": 0.4}, n_samples=2)
        s.set_question("q?")
        r = s.score(_claim(), DOCS)
        assert s.n_empty_samples == 1
        assert r.score == pytest.approx(0.6)
        assert len(seen) == 1

    def test_all_samples_empty_scores_zero(self, sc_factory):
        s, _, _ = sc_factory([""], {}, n_samples=2)
        s.set_question("q?")
        assert s.score(_claim(), DOCS).score == 0.0
        assert s.n_empty_samples == 2

    def test_alignment_none_skips_llm(self, sc_factory):
        s, calls, _ = sc_factory(["a"], {"a": 0.0})
        r = s.score(_claim("none"), DOCS)
        assert r.score == 0.0 and r.cost_usd == 0.0 and calls["n"] == 0

    def test_question_fallback_counted(self, sc_factory):
        s, _, _ = sc_factory(["a"], {"a": 0.0}, n_samples=1)
        s.score(_claim(), DOCS)
        assert s.n_fallback_question == 1

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            SelfCheckGPTScorer(nli=lambda p: [], price_in_per_mtok=0, price_out_per_mtok=0)


# ── provider selection / accounting helpers ───────────────────────────────────

from blocks import baselines as _b  # noqa: E402


class TestProviders:
    def test_fireworks_defaults(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test")
        s = SelfCheckGPTScorer(provider="fireworks", nli=lambda p: [], price_in_per_mtok=0,
                               price_out_per_mtok=0)
        assert s.base_url == "https://api.fireworks.ai/inference/v1"
        assert s.model == "accounts/fireworks/models/glm-5p3-flash"
        assert s.config["provider"] == "fireworks"

    def test_fireworks_missing_key_names_the_right_env(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
            RagasScorer(provider="fireworks", price_in_per_mtok=0, price_out_per_mtok=0)

    def test_unknown_provider(self):
        with pytest.raises(ValueError):
            RagasScorer(provider="nope", price_in_per_mtok=0, price_out_per_mtok=0)

    def test_retries_counted_from_http_requests(self):
        u = _b._Usage()
        u.add("http_requests", 3)
        u.add("calls", 1)
        assert u.last_retries == 2

    def test_retry_hook_counts_real_http_requests(self, monkeypatch):
        """A 429 then 200: the client retries once; our httpx hook sees 2 requests for 1 call."""
        monkeypatch.setenv("GROQ_API_KEY", "test")
        n = {"i": 0}

        def handler(request):
            n["i"] += 1
            if n["i"] == 1:
                return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rl"}})
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "sample"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        class PatchedClient(httpx.Client):  # subclass: openai checks isinstance(http_client, httpx.Client)
            def __init__(self, *a, **kw):
                kw["transport"] = httpx.MockTransport(handler)
                super().__init__(*a, **kw)

        monkeypatch.setattr(httpx, "Client", PatchedClient)
        s = SelfCheckGPTScorer(n_samples=1, nli=lambda p: [0.0] * len(p),
                               price_in_per_mtok=0, price_out_per_mtok=0)
        s.set_question("q?")
        s.score(_claim(), DOCS)
        assert s.last_usage["calls"] == 1 and s.last_usage["http_requests"] == 2
        assert s.last_retries == 1

    def test_selfcheck_nli_accounting_separate(self, sc_factory):
        s, _, _ = sc_factory(["a", "b"], {"a": 0.0, "b": 0.0}, n_samples=2)
        s.set_question("q?")
        s.score(_claim(), DOCS)
        assert s.last_nli["pairs"] == 2 and s.total_nli["pairs"] == 2
        assert s.last_status == "ok"


def test_export_bundle_id_is_bare_response_id():
    # harness joins score-export bundle_id on str(row["id"]) (e.g. "12003"), not the bundle file name
    from tools.export_scores import response_id
    assert response_id("ragtruth-12003") == "12003"
    assert response_id("12003") == "12003"


def test_call_record_keeps_judge_audit_fields():
    from types import SimpleNamespace as NS
    from blocks.baselines import _call_record
    resp = NS(model="mistralai/mistral-large-4-1t-a52b",
              usage=NS(prompt_tokens=10, completion_tokens=30,
                       completion_tokens_details=NS(reasoning_tokens=25)),
              choices=[NS(finish_reason="stop", message=NS(content='{"statements": []}'))])
    r = _call_record(resp, {"response_format": {"type": "json_object"}}, 12.34)
    assert r["model_returned"] == "mistralai/mistral-large-4-1t-a52b"
    assert r["response_format"] == "json_object" and r["reasoning_tokens"] == 25
    assert r["json_valid"] and r["finish_reason"] == "stop" and r["content"] == '{"statements": []}'
    bad = NS(model="m", usage=None, choices=[NS(finish_reason="length", message=NS(content=""))])
    assert _call_record(bad, {}, 1.0)["json_valid"] is False


def test_frozen_claims_must_match_manifest(tmp_path):
    import hashlib
    import json
    import pytest
    from tools.run_baselines import load_frozen_claims
    rows = [{"claim_id": "ragtruth-1-c1", "excluded": False, "y_true": 1},
            {"claim_id": "ragtruth-1-c2", "excluded": True, "y_true": 0}]
    p = tmp_path / "claims.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    man = {"name": "calib_v1.1", "gold_rules_version": "1.1", "segmenter_rules_version": "1.0",
           "outputs": {"claims.jsonl": hashlib.sha256(p.read_bytes()).hexdigest()}}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    claims, ident = load_frozen_claims(p)
    assert [c["claim_id"] for c in claims] == ["ragtruth-1-c1"] and ident["gold_rules_version"] == "1.1"
    p.write_text(p.read_text().replace('"y_true": 1', '"y_true": 0'))  # tampered gold
    with pytest.raises(SystemExit):
        load_frozen_claims(p)


def test_runner_records_invalid_output_as_null_not_zero_and_does_not_retry(tmp_path, monkeypatch):
    """judge_nan regression (Emma 2026-11-28): an invalid judge output is never a 0 score."""
    import json
    import sys
    from types import SimpleNamespace as NS
    import tools.run_baselines as rb

    class FakeJudge:
        name = "ragas-llm-judge"
        config = {"method": "ragas-llm-judge"}

        def __init__(self):
            self.n = 0
            self.last_status, self.last_invalid_reason = "ok", None

        def set_question(self, q):
            pass

        def score(self, claim, docs):
            self.n += 1
            bad = self.n == 1                       # the first claim: judge returns no statements
            self.last_status = "invalid_output" if bad else "ok"
            self.last_invalid_reason = "judge returned no statements" if bad else None
            return NS(score=float("nan") if bad else 1.0, latency_ms=1.0, cost_usd=0.0)

    judges = []
    monkeypatch.setattr(rb, "make_scorer", lambda *a, **k: judges.append(FakeJudge()) or judges[-1])
    monkeypatch.setattr(rb, "load_env", lambda: None)
    root = Path(__file__).resolve().parent.parent
    argv = ["run_baselines", "--bundles", str(root / "examples"),
            "--gold", str(root / "examples" / "answer_bundle_gold.json"), "--dataset-version", "t",
            "--out", str(tmp_path), "--scorers", "ragas", "--runs", "1",
            "--price-in", "0", "--price-out", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    rb.main()
    rows = [json.loads(l) for l in (tmp_path / "scores.jsonl").read_text().splitlines()]
    bad = [r for r in rows if r["status"] == "invalid_output"]
    assert len(bad) == 1 and bad[0]["score"] is None and bad[0]["error"].startswith("invalid_output")
    s = json.loads((tmp_path / "summary.json").read_text())["scorers"]["ragas-llm-judge"]
    assert s["n_invalid_output_rows"] == 1 and s["complete"] is True
    assert s["per_run"][0]["n_invalid_output"] == 1 and s["per_run"][0]["coverage"] < 1
    rb.main()                                   # resume: the invalid row is final, not re-asked
    assert judges[-1].n == 0


def test_pilot_report_relabels_legacy_judge_nan_and_never_scores_it(tmp_path):
    import json
    from tools.pilot_report import report
    rows = [  # 2 runs x 3 claims; c3 was logged as judge_nan with score 0.0 in run 1
        {"method": "m", "run": 1, "claim_id": "c1", "y_true": 1, "score": 0.0, "status": "ok", "latency_ms": 1},
        {"method": "m", "run": 1, "claim_id": "c2", "y_true": 0, "score": 1.0, "status": "ok", "latency_ms": 1},
        {"method": "m", "run": 1, "claim_id": "c3", "y_true": 0, "score": 0.0, "status": "judge_nan", "latency_ms": 1},
        {"method": "m", "run": 2, "claim_id": "c1", "y_true": 1, "score": 0.0, "status": "ok", "latency_ms": 1},
        {"method": "m", "run": 2, "claim_id": "c2", "y_true": 0, "score": 1.0, "status": "ok", "latency_ms": 1},
        {"method": "m", "run": 2, "claim_id": "c3", "y_true": 0, "score": 0.0, "status": "judge_nan", "latency_ms": 1},
    ]
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "r", "dataset": {}, "scorers": {}, "code": {}}))
    r = report(tmp_path)
    assert len(r["relabelled_legacy_judge_nan"]) == 2
    m = r["per_run"][0]
    assert m["n_scored"] == 2 and m["false_unfaithful"]["n"] == 0 and m["gold_agreement"] == 1.0
    assert r["determinism"]["claims_ok_in_every_run"] == 2          # c3 is not counted as "stable"
    assert "c3" in r["determinism"]["claims_not_ok_in_every_run"]
    assert '"judge_nan"' in (tmp_path / "scores.jsonl").read_text()  # the log itself is untouched
