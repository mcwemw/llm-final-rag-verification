"""Gold rules v1.1 + calib_v1.1 evidence checks (offline). Reference-level checks always run;
text-dependent checks use data/hydrated/ and skip, with a reason, without local RAGTruth."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold_rules import claims_for, gold_for_v11  # noqa: E402
from build_calib_subset import gold_for as gold_for_v10  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# The repo keeps references only (tools/hydrate.py); text-dependent checks use the hydrated copies.
F10, F11 = ROOT / "data" / "frozen" / "calib_v1", ROOT / "data" / "frozen" / "calib_v1.1"
V10, V11 = ROOT / "data" / "hydrated" / "calib_v1", ROOT / "data" / "hydrated" / "calib_v1.1"
needs_hydrated = pytest.mark.skipif(
    not ((V10 / "claims.jsonl").exists() and (V11 / "claims.jsonl").exists()),
    reason="run `python3 tools/hydrate.py --all` first (needs local RAGTruth)")


def rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _refs_ok(f):
    """Repo side: manifest unchanged since migration, only manifest/STATS/storage/refs tracked."""
    snap = json.loads((ROOT / "data" / "migration_evidence" / f.name / "pre_migration_sha256.json").read_text())
    assert hashlib.sha256((f / "manifest.json").read_bytes()).hexdigest() == snap["files"]["manifest.json"]["sha256"]
    storage = json.loads((f / "storage.json").read_text())
    for rel, digest in storage["refs"].items():
        assert hashlib.sha256((f / rel).read_bytes()).hexdigest() == digest, rel
    tracked = subprocess.run(["git", "ls-files", str(f)], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if tracked:  # skip outside a git checkout
        rel_tracked = {str(Path(x).relative_to(f.relative_to(ROOT))) for x in tracked}
        assert rel_tracked == {"manifest.json", "STATS.md", "storage.json"} | set(storage["refs"])
    return json.loads((f / "manifest.json").read_text())


def _hydrated_ok(f, h):
    m = json.loads((f / "manifest.json").read_text())
    for rel, digest in m["outputs"].items():
        assert hashlib.sha256((h / rel).read_bytes()).hexdigest() == digest, rel
    return m


def test_punctuation_only_overlap_is_not_a_positive():
    resp = "It measures gases. A higher footprint warms."
    claim = {"char_start": 0, "char_end": 18}            # "It measures gases."
    span = [{"start": 17, "end": 44, "label_type": "Subtle Baseless Info"}]   # starts at the "."
    assert gold_for_v10(claim, span)["y_true"] == 1
    assert gold_for_v11(claim, span, resp)["y_true"] == 0
    word = [{"start": 12, "end": 30, "label_type": "Evident Conflict"}]       # covers "gases"
    assert gold_for_v11(claim, word, resp)["gold_verdict"] == "FAIL"


def test_calib_v1_is_untouched():
    _refs_ok(F10)


@needs_hydrated
def test_calib_v1_hydrates_byte_identical():
    _hydrated_ok(F10, V10)


@needs_hydrated
def test_v10_path_reproduces_frozen_calib_v1_claims():
    got = [c for r in rows(V10 / "responses.jsonl") for c in claims_for(r, "1.0")]
    text = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in got)
    assert text == (V10 / "claims.jsonl").read_text(encoding="utf-8")


def test_calib_v1_1_evidence():
    """From the references alone (no corpus text needed)."""
    m = _refs_ok(F11)
    assert m["parent"]["manifest_sha256"] == hashlib.sha256((F10 / "manifest.json").read_bytes()).hexdigest()
    assert m["gold_rules_version"] == "1.1" and m["segmenter_rules_version"] == "1.0"
    assert m["outputs"]["responses.jsonl"] == json.loads((F10 / "manifest.json").read_text())["outputs"]["responses.jsonl"]
    assert rows(F11 / "refs" / "responses.ref.jsonl") == rows(F10 / "refs" / "responses.ref.jsonl")
    old = {c["claim_id"]: c for c in rows(F10 / "refs" / "claims.ref.jsonl")}
    diff = [c["claim_id"] for c in rows(F11 / "refs" / "claims.ref.jsonl")
            if (c["gold_verdict"], c["y_true"]) != (old[c["claim_id"]]["gold_verdict"], old[c["claim_id"]]["y_true"])]
    assert diff == m["label_changes_vs_parent"] == ["ragtruth-17279-c2"]


@needs_hydrated
def test_calib_v1_1_hydrates_byte_identical_including_label_changes():
    m = _hydrated_ok(F11, V11)
    assert "LABEL_CHANGES.md" in m["outputs"]
    assert (V11 / "responses.jsonl").read_bytes() == (V10 / "responses.jsonl").read_bytes()
