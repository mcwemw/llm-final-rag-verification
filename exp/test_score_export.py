"""Unit tests for exp/score_export.py (score-export-v1 ingest).
Run: python3 exp/test_score_export.py   (or pytest)

Covers the fail-loud contract: missing fields, missing/invalid direction,
wrong schema_version, bundle_id must be a pure response id, direction flip,
null-score coverage, selfcheck determinism honesty.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_export import (  # noqa: E402
    load_export, to_hallucination_prob, coverage, determinism_trusted,
    response_verdicts, summarize, ScoreExportError,
)


def _valid(**over):
    obj = {
        "schema_version": "score-export-v1",
        "run_id": "2026-11-27T01-30_ragas_llama70b",
        "technique": "ragas",
        "model": "llama-3.3-70b",
        "scorer_version": "baselines.py@abc123",
        "dataset_version": "calib_v1",
        "dataset_sha": "3326b9a6",
        "score_direction": "high_means_faithful",
        "score_range": [0.0, 1.0],
        "rerun": False,
        "created_at": "2026-11-27T01:30:00+08:00",
        "scores": [
            {"bundle_id": "12003", "claim_id": "ragtruth-12003-c2", "score": 0.9, "error": None},
            {"bundle_id": "12007", "claim_id": "ragtruth-12007-c1", "score": 0.1, "error": None},
        ],
    }
    obj.update(over)
    return obj


def _expect(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} but none raised")


def test_valid_loads():
    obj = load_export(_valid())
    assert obj["dataset_version"] == "calib_v1"


def test_wrong_schema_version_rejected():
    _expect(ScoreExportError, load_export, _valid(schema_version="v2"))


def test_missing_required_field_rejected():
    bad = _valid()
    del bad["dataset_version"]
    _expect(ScoreExportError, load_export, bad)


def test_missing_direction_rejected():
    bad = _valid()
    del bad["score_direction"]
    _expect(ScoreExportError, load_export, bad)


def test_invalid_direction_rejected():
    _expect(ScoreExportError, load_export, _valid(score_direction="higher_better"))


def test_bad_score_range_rejected():
    _expect(ScoreExportError, load_export, _valid(score_range=[1.0, 0.0]))
    _expect(ScoreExportError, load_export, _valid(score_range=[0.0]))


def test_empty_scores_rejected():
    _expect(ScoreExportError, load_export, _valid(scores=[]))


def test_selfcheck_requires_extra_fields():
    # a selfcheck export missing n_samples/generator_model/samples_reset is rejected
    _expect(ScoreExportError, load_export, _valid(technique="selfcheck"))
    ok = _valid(technique="selfcheck", n_samples=3,
                generator_model="llama-3.3-70b", samples_reset=True)
    assert load_export(ok)["technique"] == "selfcheck"


def test_bundle_id_must_be_pure_response_id():
    # decorated ids rejected (the matching-bug guard)
    for bad_id in ["bundle_03", "ragtruth-12003", "12003-c2", "resp12003"]:
        bad = _valid()
        bad["scores"][0]["bundle_id"] = bad_id
        _expect(ScoreExportError, to_hallucination_prob, bad)


def test_bundle_id_universe_mismatch_rejected():
    # a pure id that is not among the loaded dataset ids is rejected
    _expect(ScoreExportError, to_hallucination_prob, _valid(),
            {"99999", "88888"})  # universe excludes 12003/12007


def test_bundle_id_universe_match_ok():
    rows = to_hallucination_prob(_valid(), {"12003", "12007"})
    assert len(rows) == 2


def test_direction_flip_high_means_faithful():
    # faithful-high: score 0.9 → low hallucination prob
    rows = to_hallucination_prob(_valid(score_direction="high_means_faithful"))
    assert abs(rows[0]["p_hallu"] - 0.1) < 1e-9   # 1 - 0.9
    assert abs(rows[1]["p_hallu"] - 0.9) < 1e-9   # 1 - 0.1


def test_direction_flip_high_means_hallucinated():
    rows = to_hallucination_prob(_valid(score_direction="high_means_hallucinated"))
    assert abs(rows[0]["p_hallu"] - 0.9) < 1e-9   # score IS hallu prob
    assert abs(rows[1]["p_hallu"] - 0.1) < 1e-9


def test_score_out_of_range_rejected():
    bad = _valid()
    bad["scores"][0]["score"] = 1.5
    _expect(ScoreExportError, to_hallucination_prob, bad)


def test_null_score_is_coverage_gap_not_zero():
    obj = _valid()
    obj["scores"][0]["score"] = None
    obj["scores"][0]["error"] = "groq_timeout"
    rows = to_hallucination_prob(obj)
    assert rows[0]["p_hallu"] is None          # NOT coerced to 0.0
    assert abs(coverage(rows) - 0.5) < 1e-9     # 1 of 2 scored


def test_missing_score_field_rejected():
    bad = _valid()
    del bad["scores"][0]["error"]
    _expect(ScoreExportError, to_hallucination_prob, bad)


def test_determinism_untrusted_when_samples_not_reset():
    sc = _valid(technique="selfcheck", n_samples=3,
                generator_model="llama-3.3-70b", rerun=True, samples_reset=False)
    assert determinism_trusted(sc) is False
    sc2 = _valid(technique="selfcheck", n_samples=3,
                 generator_model="llama-3.3-70b", rerun=True, samples_reset=True)
    assert determinism_trusted(sc2) is True


def test_response_verdicts_any_claim_flags():
    # one response, two claims; OR over claims with p_hallu >= threshold
    obj = _valid(score_direction="high_means_hallucinated")
    obj["scores"] = [
        {"bundle_id": "12003", "claim_id": "ragtruth-12003-c1", "score": 0.1, "error": None},
        {"bundle_id": "12003", "claim_id": "ragtruth-12003-c2", "score": 0.8, "error": None},
        {"bundle_id": "12007", "claim_id": "ragtruth-12007-c1", "score": 0.2, "error": None},
    ]
    v = response_verdicts(to_hallucination_prob(obj), threshold=0.5)
    assert v["12003"] == 1   # c2 crosses
    assert v["12007"] == 0   # below


def test_response_with_only_errored_claims_is_uncovered():
    obj = _valid(score_direction="high_means_hallucinated")
    obj["scores"] = [
        {"bundle_id": "12003", "claim_id": "ragtruth-12003-c1", "score": None, "error": "timeout"},
    ]
    v = response_verdicts(to_hallucination_prob(obj), threshold=0.5)
    assert "12003" not in v   # cannot judge → not faked as 0


def test_response_verdicts_bad_threshold_rejected():
    rows = to_hallucination_prob(_valid())
    _expect(ScoreExportError, response_verdicts, rows, 1.5)


def test_summarize_reports_coverage_and_counts():
    s = summarize(_valid(score_direction="high_means_hallucinated"), {"12003", "12007"}, 0.5)
    assert s["n_claims"] == 2 and s["coverage"] == 1.0
    assert s["dataset_version"] == "calib_v1"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items()
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
