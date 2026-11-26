"""Acceptance tests for score_export (score-export-v1 validator). Stdlib only, no network.
Run: python exp/test_score_export.py   (or pytest). Maps to HARNESS-ACCEPTANCE #5/#6/#7.
"""
from __future__ import annotations
import copy
try:
    from score_export import (validate, to_hallucination_prob, coverage,
                              determinism_trustworthy, ScoreExportError)
except ImportError:
    from exp.score_export import (validate, to_hallucination_prob, coverage,
                                 determinism_trustworthy, ScoreExportError)


def _valid_selfcheck() -> dict:
    return {
        "schema_version": "score-export-v1", "run_id": "r1", "technique": "selfcheck",
        "model": "deberta-v3-large", "scorer_version": "baselines.py@abc",
        "dataset_version": "calib_v1", "dataset_sha": "deadbeef",
        "n_samples": 3, "generator_model": "openai/gpt-oss-20b", "temperature": 1.0,
        "score_direction": "high_means_faithful", "score_range": [0.0, 1.0],
        "rerun": False, "samples_reset": True, "created_at": "2026-11-27T01:30:00+08:00",
        "scores": [
            {"bundle_id": "b3", "claim_id": "c1", "score": 0.21, "error": None},
            {"bundle_id": "b7", "claim_id": "c1", "score": None, "error": "groq_timeout"},
        ],
    }


def _assert_raises(fn, substr=""):
    try:
        fn()
    except ScoreExportError as e:
        assert substr in str(e), f"wrong error: {e!r} (want {substr!r})"
        return
    raise AssertionError(f"expected ScoreExportError containing {substr!r}, none raised")


def test_valid_passes():
    validate(_valid_selfcheck())


def test_schema_version_mismatch_failloud():          # acceptance #7
    bad = _valid_selfcheck(); bad["schema_version"] = "score-export-v2"
    _assert_raises(lambda: validate(bad), "schema_version")


def test_score_direction_required():                   # acceptance #5
    bad = _valid_selfcheck(); del bad["score_direction"]
    _assert_raises(lambda: validate(bad), "missing required")
    bad2 = _valid_selfcheck(); bad2["score_direction"] = "higher_better"
    _assert_raises(lambda: validate(bad2), "score_direction")


def test_selfcheck_requires_samples_reset():           # acceptance #6
    bad = _valid_selfcheck(); del bad["samples_reset"]
    _assert_raises(lambda: validate(bad), "samples_reset")


def test_null_score_must_carry_error():                # §3.3 errors not swallowed
    bad = _valid_selfcheck(); bad["scores"][1]["error"] = None
    _assert_raises(lambda: validate(bad), "swallow")


def test_score_out_of_range():
    bad = _valid_selfcheck(); bad["scores"][0]["score"] = 1.7
    _assert_raises(lambda: validate(bad), "out of score_range")


def test_direction_flip_normalization():               # §4
    f = _valid_selfcheck()  # high_means_faithful: score 0.21 -> p_hallu 0.79
    out = to_hallucination_prob(f)
    assert abs(out[0]["p_hallu"] - 0.79) < 1e-9, out[0]
    assert out[1]["p_hallu"] is None                   # null stays null
    h = _valid_selfcheck(); h["score_direction"] = "high_means_hallucinated"
    out2 = to_hallucination_prob(h)                    # score 0.21 -> p_hallu 0.21
    assert abs(out2[0]["p_hallu"] - 0.21) < 1e-9, out2[0]


def test_coverage_counts_null_against_not_as_zero():   # §3.3
    assert abs(coverage(_valid_selfcheck()) - 0.5) < 1e-9   # 1 of 2 non-null


def test_determinism_trust():                          # §3.4
    f = _valid_selfcheck()
    assert determinism_trustworthy(f) is False         # rerun False
    f["rerun"] = True
    assert determinism_trustworthy(f) is True          # rerun + samples_reset
    f["samples_reset"] = False
    assert determinism_trustworthy(f) is False         # rerun but no reset -> untrustworthy


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t(); print(f"PASS {t.__name__}"); passed += 1
    print(f"\n{passed}/{len(tests)} passed")
