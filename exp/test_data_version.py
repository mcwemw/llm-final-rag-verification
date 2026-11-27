"""Unit tests for exp/data_version.py (dataset + thresholds binding, fail-loud).
Run: python3 exp/test_data_version.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_version as dv  # noqa: E402


def _expect(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} but none raised")


def test_verify_absent_is_not_faked_as_verified():
    # a dir with no data files → status 'absent', not 'verified', no raise
    with tempfile.TemporaryDirectory() as d:
        res = dv.verify(d, strict=True)
    assert res["status"] == "absent"
    assert res["version"] == dv.DATASET_VERSION


def test_verify_mismatch_raises():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "response.jsonl"), "w") as f:
            f.write("not the pinned bytes\n")
        _expect(dv.DataVersionError, dv.verify, d, True)


def test_verify_mismatch_nonstrict_reports_not_raises():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "response.jsonl"), "w") as f:
            f.write("not the pinned bytes\n")
        res = dv.verify(d, strict=False)
    assert res["status"] == "mismatch"
    assert "response.jsonl" in res["mismatches"]


def test_load_thresholds_has_calib_binding():
    th = dv.load_thresholds(verify_manifest=True)
    assert th["calib_version"] == "calib_v1"
    assert len(th["calib_manifest_sha256"]) == 64


def test_load_thresholds_missing_binding_raises():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "thresholds.json")
        with open(p, "w") as f:
            json.dump({"single_threshold_maxF1": {"t": 0.5}}, f)
        _expect(dv.DataVersionError, dv.load_thresholds, p, False)


def test_load_thresholds_manifest_sha_mismatch_raises():
    # point calib_path at a temp manifest whose sha != declared
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "calib"), exist_ok=True)
        with open(os.path.join(d, "calib", "manifest.json"), "w") as f:
            f.write('{"name":"tampered"}')
        p = os.path.join(d, "thresholds.json")
        with open(p, "w") as f:
            json.dump({"calib_version": "calib_v1",
                       "calib_manifest_sha256": "0" * 64,
                       "calib_path": "calib"}, f)
        # make REPO_ROOT resolve to d for this call
        old = dv.REPO_ROOT
        dv.REPO_ROOT = d
        try:
            _expect(dv.DataVersionError, dv.load_thresholds, p, True)
        finally:
            dv.REPO_ROOT = old


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items()
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
