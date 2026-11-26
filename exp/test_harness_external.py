"""Integration test: run_harness.score_external consumes a score-export-v1 JSON and
produces a scored table row. Synthetic rows (no RAGTruth data) + temp file, no network.
Run with a numpy-capable python:
  ~/.local/pipx/venvs/skill-seekers/bin/python exp/test_harness_external.py
"""
from __future__ import annotations
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import run_harness


def _rows():
    # 8 responses, ids r0..r7; y_true: faithful=0, hallucinated=1 (balanced).
    out = []
    for i in range(8):
        y = i % 2  # 0,1,0,1,...
        out.append({"id": f"r{i}", "source_id": f"s{i}", "task_type": "QA",
                    "source_text": "ctx", "response": "ans", "y_true": y, "n_spans": y})
    return out


def _export(rows, path):
    # high_means_faithful: faithful rows high score, hallucinated low -> perfect separation.
    scores = []
    for r in rows:
        s = 0.1 if r["y_true"] == 1 else 0.9
        scores.append({"bundle_id": r["id"], "claim_id": "c1", "score": s, "error": None})
    # one null+error row (coverage < 1); id present but score unknown
    scores.append({"bundle_id": "r0", "claim_id": "c2", "score": None, "error": "timeout"})
    obj = {
        "schema_version": "score-export-v1", "run_id": "test-ext-1", "technique": "ragas",
        "model": "openai/gpt-oss-20b", "scorer_version": "test@1",
        "dataset_version": "calib_v1", "dataset_sha": "abc",
        "score_direction": "high_means_faithful", "score_range": [0.0, 1.0],
        "rerun": False, "created_at": "2026-11-27T01:00:00+08:00", "scores": scores,
    }
    with open(path, "w") as f:
        json.dump(obj, f)


def test_score_external_produces_row():
    rows = _rows()
    y = np.array([r["y_true"] for r in rows])
    # cal/eval masks that each contain both classes
    cal_mask = np.array([True, True, True, True, False, False, False, False])
    ev = ~cal_mask
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ext.json")
        _export(rows, p)
        row = run_harness.score_external(p, rows, cal_mask, ev, y, expected_dsv="calib_v1")
    assert row is not None, "expected a result row"
    assert row["f1"] == 1.0, f"perfect separation should give F1=1.0, got {row['f1']}"
    assert "external" in row["name"] and "ragas" in row["name"]
    assert "matched 8/8" in row["note"], row["note"]
    assert "coverage=0.89" in row["note"], row["note"]  # 8 of 9 claims non-null


def test_dataset_version_mismatch_failloud():
    rows = _rows(); y = np.array([r["y_true"] for r in rows])
    cal_mask = np.array([True]*4 + [False]*4); ev = ~cal_mask
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ext.json"); _export(rows, p)
        try:
            run_harness.score_external(p, rows, cal_mask, ev, y, expected_dsv="calib_v2")
        except run_harness.score_export.ScoreExportError as e:
            assert "dataset_version mismatch" in str(e); return
    raise AssertionError("expected dataset_version mismatch to raise")


def test_unmatched_ids_skipped():
    rows = _rows(); y = np.array([r["y_true"] for r in rows])
    cal_mask = np.array([True]*4 + [False]*4); ev = ~cal_mask
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ext.json")
        obj = {"schema_version": "score-export-v1", "run_id": "x", "technique": "nli",
               "model": "m", "scorer_version": "v", "dataset_version": "calib_v1",
               "score_direction": "high_means_hallucinated", "score_range": [0.0, 1.0],
               "rerun": False, "created_at": "t",
               "scores": [{"bundle_id": "ZZZ", "claim_id": "c1", "score": 0.5, "error": None}]}
        with open(p, "w") as f: json.dump(obj, f)
        row = run_harness.score_external(p, rows, cal_mask, ev, y)
    assert row is None, "unmatched ids should skip (return None), not fabricate a row"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
