"""Dataset-version binding for the harness (fail-loud).

The frozen calibration subset (data/frozen/calib_v1) was built from a specific
RAGTruth snapshot. Its manifest pins that snapshot by sha256. The harness must
run its eval on the SAME snapshot, otherwise thresholds calibrated on calib_v1
would be applied to different data.

This module pins those sha256 values and verifies the harness's local
exp/data/*.jsonl against them. exp/data/ is gitignored (RAGTruth is fetched,
not vendored), so:
  - files present  -> sha MUST match the pins, else raise (fail-loud)
  - files absent   -> skip with a clear note (CI has no data); never silently pass as verified
"""
from __future__ import annotations

import hashlib
import os

# Pinned from data/frozen/calib_v1/manifest.json -> "inputs"
# (RAGTruth commit c103204b9ce28d6bbad859304bf30de72b8ed8fe)
DATASET_VERSION = "ragtruth@c103204"
PINS = {
    "response.jsonl": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "source_info.jsonl": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class DataVersionError(RuntimeError):
    """Raised when local data does not match the pinned RAGTruth snapshot."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(data_dir: str = DATA_DIR, strict: bool = True) -> dict:
    """Verify exp/data/*.jsonl against the pinned snapshot.

    Returns {status, version, checked, mismatches}. status is one of:
      "verified"  — every present file matched its pin
      "absent"    — no data files present (CI); nothing to verify
    strict=True (default): a sha MISMATCH raises DataVersionError.
    A mismatch is always fatal; absence is not (but is reported, not hidden)."""
    checked, mismatches, present = {}, [], 0
    for name, want in PINS.items():
        p = os.path.join(data_dir, name)
        if not os.path.exists(p):
            checked[name] = "absent"
            continue
        present += 1
        got = _sha256(p)
        checked[name] = "ok" if got == want else f"MISMATCH got={got[:12]}"
        if got != want:
            mismatches.append(name)

    if mismatches and strict:
        raise DataVersionError(
            f"dataset does not match pinned {DATASET_VERSION}: {mismatches}. "
            f"Thresholds were calibrated on this snapshot; refusing to run on "
            f"different data.")

    status = "verified" if present and not mismatches else (
        "absent" if present == 0 else "mismatch")
    return {"status": status, "version": DATASET_VERSION,
            "checked": checked, "mismatches": mismatches}


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
THRESHOLDS = os.path.join(os.path.dirname(__file__), "thresholds.json")


def load_thresholds(path: str = THRESHOLDS, verify_manifest: bool = True) -> dict:
    """Load thresholds.json and assert it is explicitly bound to a calibration
    subset (calib_version + calib_manifest_sha256). When verify_manifest and the
    referenced manifest exists, assert its sha256 matches the pinned value —
    so a threshold can never be silently applied to a different calibration set."""
    import json
    with open(path) as f:
        th = json.load(f)
    for k in ("calib_version", "calib_manifest_sha256"):
        if k not in th:
            raise DataVersionError(
                f"thresholds.json missing calib binding field {k!r}: thresholds "
                f"must declare which calibration subset they were tuned on.")
    if verify_manifest:
        manifest = os.path.join(REPO_ROOT, th.get("calib_path", "data/frozen/calib_v1"),
                                "manifest.json")
        if os.path.exists(manifest):
            got = _sha256(manifest)
            if got != th["calib_manifest_sha256"]:
                raise DataVersionError(
                    f"thresholds calib_manifest_sha256 "
                    f"{th['calib_manifest_sha256'][:12]} != actual manifest "
                    f"{got[:12]} — thresholds are bound to a different calib_v1.")
    return th


if __name__ == "__main__":
    import json
    print(json.dumps({"data": verify(strict=False),
                      "thresholds_binding": {
                          k: load_thresholds(verify_manifest=True)[k]
                          for k in ("calib_version", "calib_manifest_sha256")}},
                     indent=2))
