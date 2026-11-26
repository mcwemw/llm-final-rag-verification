"""END-TO-END #6 integration test — drives the REAL harness path
`run_harness.selfcheck_lifecycle` (set_question -> reset_samples -> score) then build_export
then run_harness.score_external (final metrics). Emma's 4 points + reset-exception honesty.
Offline (injected sampler/consistency). Run with numpy-capable python:
  ~/.local/pipx/venvs/skill-seekers/bin/python exp/test_selfcheck_e2e.py
"""
from __future__ import annotations
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import score_export
import run_harness
from scorers.selfcheck import SelfCheckScorer

# Response-level records (harness scores responses). source_id deliberately NOT in RAGTruth
# data, so selfcheck_lifecycle's qmap misses and uses each record's own question.
RECORDS = [
    {"id": "r0", "source_id": "TESTSRC0", "question": "QA", "response": "ok one",  "y": 0},
    {"id": "r1", "source_id": "TESTSRC1", "question": "QB", "response": "bad one", "y": 1},
    {"id": "r2", "source_id": "TESTSRC2", "question": "QC", "response": "ok two",  "y": 0},
    {"id": "r3", "source_id": "TESTSRC3", "question": "QD", "response": "bad two", "y": 1},
]


class _S:
    n = 0
    draws = []   # (record_id, question, draw_n)


def fake_sampler(rec, question, n, temperature):
    _S.n += 1
    _S.draws.append((str(rec.get("id")), question, _S.n))
    return [f"d{_S.n}-{i}" for i in range(n)]


def fake_consistency(response, samples):
    return 0.1 if "bad" in response else 0.9   # hallucinated('bad') low, faithful high


def _scorer():
    _S.n = 0; _S.draws = []
    return SelfCheckScorer(sampler=fake_sampler, consistency_fn=fake_consistency,
                           n_samples=2, model="M", temperature=1.0)


def test_e2e_through_harness_lifecycle():
    scorer = _scorer()
    # (2) same scorer, 3 runs — via the REAL harness function
    all_runs, resets = run_harness.selfcheck_lifecycle(RECORDS, scorer, n_runs=3)
    assert len(all_runs) == 3 and len(resets) == 3
    assert all(resets) is True

    # (1) questions did not cross-contaminate
    qmap = {"r0": "QA", "r1": "QB", "r2": "QC", "r3": "QD"}
    for rid, q, _ in _S.draws:
        assert q == qmap[rid], f"record {rid} sampled with wrong question {q!r}"

    # (3) independent sampling per rerun: each record got a distinct fresh draw in each run
    from collections import defaultdict
    per_rec = defaultdict(list)
    for rid, _, dn in _S.draws:
        per_rec[rid].append(dn)
    for rid, d in per_rec.items():
        assert len(d) == 3 and len(set(d)) == 3, f"{rid}: draws not independent per run: {d}"

    # (4a) build one export per run, honest samples_reset, validates
    exports = []
    for i, run_scores in enumerate(all_runs):
        scores = [{"bundle_id": r["id"], "claim_id": "_response", "score": float(s), "error": None}
                  for r, s in zip(RECORDS, run_scores)]
        e = score_export.build_export(technique="selfcheck", model="M",
                                      dataset_version="calib_v1", scores=scores,
                                      run_id=f"run{i}", rerun=(i > 0), samples_reset=resets[i],
                                      n_samples=2, generator_model="M")
        assert e["samples_reset"] is True
        assert score_export.determinism_trustworthy(e) is bool(e["rerun"])
        exports.append(e)

    # (4b) harness final metrics compute on a real export
    rows = [{"id": r["id"], "source_id": r["id"], "task_type": "QA", "source_text": "c",
             "response": "x", "y_true": r["y"], "n_spans": r["y"]} for r in RECORDS]
    y = np.array([r["y"] for r in RECORDS])
    cal_mask = np.array([True, True, False, False]); ev = ~cal_mask
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run0.json")
        with open(p, "w") as f:
            json.dump(exports[0], f)
        row = run_harness.score_external(p, rows, cal_mask, ev, y, expected_dsv="calib_v1")
    assert row is not None and isinstance(row["f1"], float)
    assert set(row["claim_scores"]) == {"r0", "r1", "r2", "r3"}   # req A claim-level retained


def test_reset_exception_makes_run_untrusted():
    """(3) reset_samples raising on a rerun => that run's flag False => determinism not trusted."""
    scorer = _scorer()
    calls = {"n": 0}
    orig = scorer.reset_samples

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:            # fail on the 2nd run's reset
            raise RuntimeError("reset failed")
        orig()
    scorer.reset_samples = flaky

    all_runs, resets = run_harness.selfcheck_lifecycle(RECORDS, scorer, n_runs=3)
    assert resets == [True, False, True], resets
    # the failed run must not yield trustworthy determinism
    e = score_export.build_export(technique="selfcheck", model="M", dataset_version="calib_v1",
                                  scores=[{"bundle_id": "r0", "claim_id": "_response",
                                           "score": 0.5, "error": None}],
                                  run_id="run1", rerun=True, samples_reset=resets[1],
                                  n_samples=2, generator_model="M")
    assert score_export.determinism_trustworthy(e) is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
