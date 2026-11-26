"""Proves set_question() + reset_samples() have real teeth (Jimmy #6).
Offline: inject a fake sampler + consistency_fn (no Groq, no transformers).
Run: ~/.local/pipx/venvs/skill-seekers/bin/python exp/test_scorer_interface.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from scorers.selfcheck import SelfCheckScorer


class _Draws:
    """Each fresh sampler call returns a NEW draw id -> samples differ between draws."""
    n = 0
    last_question = None


def fake_sampler(rec, question, n, temperature):
    _Draws.n += 1
    _Draws.last_question = question
    return [f"sample-{_Draws.n}-{i}" for i in range(n)]


def fake_consistency(response, samples):
    draw = int(samples[0].split("-")[1])    # which draw produced these samples
    return (draw % 7) / 7.0                  # different draw -> (generally) different score


def _scorer():
    _Draws.n = 0
    return SelfCheckScorer(sampler=fake_sampler, consistency_fn=fake_consistency, n_samples=3)


def test_reuse_without_reset_is_fake_determinism():
    sc = _scorer()
    rec = {"id": "r1", "response": "x"}
    s1 = sc.score(rec)            # draw 1, cached
    s2 = sc.score(rec)            # reuses cached draw 1
    assert s1 == s2, "without reset, cached samples must be reused (so rerun looks identical)"


def test_reset_samples_actually_resamples():
    sc = _scorer()
    rec = {"id": "r1", "response": "x"}
    s1 = sc.score(rec)           # draw 1 -> 1/7
    sc.reset_samples()
    s2 = sc.score(rec)           # draw 2 -> 2/7 (fresh samples)
    assert s1 != s2, "reset_samples() must force an independent re-draw (not a no-op)"


def test_set_question_overrides_record_and_is_used():
    sc = _scorer()
    sc.set_question("Q-FROM-BUNDLE")
    sc.score({"id": "r9", "response": "y", "question": "Q-from-record"})
    assert _Draws.last_question == "Q-FROM-BUNDLE", \
        "set_question(bundle[q]) must take precedence over the record's question"
    assert sc.question == "Q-FROM-BUNDLE"


def test_samples_reset_flag_lifecycle():
    sc = _scorer()
    assert sc.samples_reset is True           # fresh scorer: reset
    sc.score({"id": "a", "response": "z"})
    assert sc.samples_reset is False          # a scoring pass consumed the reset state
    sc.reset_samples()
    assert sc.samples_reset is True           # reset again


def test_reset_clears_all_cached_records():
    sc = _scorer()
    sc.score({"id": "a", "response": "z"})
    sc.score({"id": "b", "response": "z"})
    assert len(sc._sample_cache) == 2
    sc.reset_samples()
    assert len(sc._sample_cache) == 0, "reset must drop ALL cached samples"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
