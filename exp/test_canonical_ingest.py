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


def test_determinism_trusted_only_on_reruns():
    # release semantics: rerun=false -> determinism not measured -> False
    for p in FIX:
        obj = _load(p)
        assert se.determinism_trusted(obj) is bool(obj["rerun"]), os.path.basename(p)
        if obj["technique"] == "selfcheck":
            assert obj["samples_reset"] is True


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


# ---- run_harness external entry (restored; goes through the canonical module) ----

def _harness_rows():
    """Synthetic harness rows: the 3 pilot-2 responses + 5 others, alternating labels."""
    import numpy as np
    ids = ["12003", "12007", "12091", "13051", "13639", "13798", "13809", "13847"]
    rows = [{"id": i, "y_true": k % 2} for k, i in enumerate(ids)]
    y = np.array([r["y_true"] for r in rows])
    cal = np.array([True, False, True, False, True, False, True, False])
    return rows, cal, ~cal, y


@pytest.mark.parametrize("path", FIX, ids=lambda p: os.path.basename(p))
def test_score_external_consumes_every_pilot2_export(path):
    import run_harness
    rows, cal, ev, y = _harness_rows()
    r = run_harness.score_external(path, rows, cal, ev, y, expected_dsv="calib_v1")
    assert r is not None and "[external]" in r["name"]
    assert "matched 3/8 rows" in r["note"] and "coverage=1.00" in r["note"]
    obj = _load(path)
    assert ("determinism=trusted" in r["note"]) is bool(obj["rerun"])
    assert sorted(r["claim_scores"]) == ["12003", "12007", "12091"]
    assert sum(len(v) for v in r["claim_scores"].values()) == 15


def test_score_external_rejects_decorated_ids_and_wrong_version(tmp_path):
    import run_harness
    rows, cal, ev, y = _harness_rows()
    obj = copy.deepcopy(_load(FIX[3]))
    obj["scores"][0]["bundle_id"] = "ragtruth-12003"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(obj))
    with pytest.raises(se.ScoreExportError):
        run_harness.score_external(str(bad), rows, cal, ev, y)
    with pytest.raises(se.ScoreExportError):
        run_harness.score_external(FIX[3], rows, cal, ev, y, expected_dsv="calib_v1.1")


def test_score_external_rejects_ids_outside_loaded_rows():
    import run_harness
    rows, cal, ev, y = _harness_rows()
    with pytest.raises(se.ScoreExportError):   # 12091 is not among the harness rows
        run_harness.score_external(FIX[0], rows[:2] + rows[3:], cal[:7], ev[:7], y[:7])


def test_harness_cli_has_score_export_flags():
    import subprocess
    out = subprocess.run([sys.executable, os.path.join(HERE, "run_harness.py"), "--help"],
                         capture_output=True, text=True).stdout
    assert "--score-export" in out and "--dataset-version" in out and "--dataset-sha" in out
