"""Canonical-ingest compatibility tests (Gate 2 exp/score_export.py is the single ingest).

Fixtures: the 9 pilot-2 exports (3 techniques x 3 reruns, calib_v1, 15 claims each), produced
by tools/export_scores.py with bare response ids. They are FORMAT fixtures only: pilot 2 used
GLM-5.3-Flash (since excluded) and gold rules v1.0, so their scores are not results.
Also covers the harness's own response-level export shape (claim_id "_response").
"""
import copy
import glob
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_export as se  # noqa: E402

FIX = sorted(glob.glob(os.path.join(HERE, "fixtures", "pilot2_exports", "*.json")))
CALIB_V1 = os.path.join(HERE, "..", "data", "frozen", "calib_v1", "responses.jsonl")


def _universe():
    with open(CALIB_V1) as f:
        return {str(json.loads(l)["id"]) for l in f if l.strip()}


def _load(p):
    with open(p) as f:
        return json.load(f)


def test_all_nine_pilot2_exports_present():
    names = [os.path.basename(p) for p in FIX]
    assert names == [f"{t}_run{k}.json" for t in ("presence", "ragas", "selfcheck") for k in (1, 2, 3)]


@pytest.mark.parametrize("path", FIX, ids=lambda p: os.path.basename(p))
def test_pilot2_export_ingests_with_full_coverage(path):
    obj = se.load_export(path, expected_dataset_version="calib_v1")
    rows = se.to_hallucination_prob(obj, _universe())
    assert len(rows) == len(obj["scores"]) == 15
    assert se.coverage(rows) == 1.0
    # high_means_faithful on [0,1]: p_hallu == 1 - score, row by row, ids unchanged
    for r, s in zip(rows, obj["scores"]):
        assert r["bundle_id"] == s["bundle_id"] and r["claim_id"] == s["claim_id"]
        assert r["p_hallu"] == pytest.approx(1.0 - s["score"])
    summ = se.summarize(obj, _universe())
    assert summ["n_claims"] == 15 and summ["n_responses_scored"] == 3


def test_selfcheck_reruns_reset_samples():
    for p in FIX:
        obj = _load(p)
        if obj["technique"] == "selfcheck":
            assert obj["samples_reset"] is True and se.determinism_trusted(obj)


@pytest.mark.parametrize("bad", ["ragtruth-12003", "bundle_03", "12003a", ""])
def test_decorated_bundle_id_rejected(bad):
    obj = copy.deepcopy(_load(FIX[0]))
    obj["scores"][0]["bundle_id"] = bad
    with pytest.raises(se.ScoreExportError):
        se.to_hallucination_prob(obj, _universe())


def test_bundle_id_outside_dataset_rejected():
    obj = copy.deepcopy(_load(FIX[0]))
    obj["scores"][0]["bundle_id"] = "99999999"
    with pytest.raises(se.ScoreExportError):
        se.to_hallucination_prob(obj, _universe())


def test_dataset_version_mismatch_rejected():
    with pytest.raises(se.ScoreExportError):
        se.load_export(FIX[0], expected_dataset_version="calib_v1.1")
    obj = dict(_load(FIX[0]), dataset_sha="abc")
    with pytest.raises(se.ScoreExportError):
        se.load_export(obj, expected_dataset_sha="def")
    se.load_export(obj, expected_dataset_sha="abc")  # match is accepted


def test_extra_fields_are_ignored_not_rejected():
    # pilot exports carry extra fields (llm_usage, cost, status, char offsets, "bundle"): allowed
    obj = _load(FIX[3])
    assert "llm_usage" in obj and "bundle" in obj["scores"][0]
    se.load_export(obj)


def test_harness_response_level_export_shape_ingests():
    # the shape exp/run_harness.py emits per run: claim_id "_response", bundle_id = str(row id)
    obj = {"schema_version": "score-export-v1", "run_id": "r", "technique": "embedding",
           "model": "BAAI/bge-m3", "scorer_version": "x", "dataset_version": "calib_v1",
           "score_direction": "high_means_faithful", "score_range": [0, 1], "rerun": False,
           "created_at": "2026-11-27T00:00:00", "scores": [
               {"bundle_id": "12003", "claim_id": "_response", "score": 0.8, "error": None},
               {"bundle_id": "12007", "claim_id": "_response", "score": None, "error": "timeout"}]}
    rows = se.to_hallucination_prob(obj, _universe())
    assert se.coverage(rows) == 0.5 and rows[1]["p_hallu"] is None
