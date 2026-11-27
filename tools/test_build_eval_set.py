"""Checks on eval_40pos_v1.1, stored as references (no corpus text in the repo).

Reference-level checks always run. Checks that need the text run on the hydrated copy
(data/hydrated/eval_40pos_v1.1, built by tools/hydrate.py) and are skipped, with a reason,
when it is absent (e.g. CI without RAGTruth).
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "frozen" / "eval_40pos_v1.1"
HYD = ROOT / "data" / "hydrated" / "eval_40pos_v1.1"
CALIB = ROOT / "data" / "frozen" / "calib_v1"
needs_hydrated = pytest.mark.skipif(not (HYD / "claims.jsonl").exists(),
                                    reason="run `python3 tools/hydrate.py eval_40pos_v1.1` first (needs local RAGTruth)")


def rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _tracked(path):
    out = subprocess.run(["git", "ls-files", str(path)], cwd=ROOT, capture_output=True, text=True).stdout.split()
    return {str(Path(x).relative_to(path.relative_to(ROOT))) for x in out}


def test_repo_keeps_only_manifest_stats_storage_and_refs():
    storage = json.loads((EVAL / "storage.json").read_text())
    tracked = _tracked(EVAL)
    if tracked:  # inside a git checkout
        assert tracked == {"manifest.json", "STATS.md", "storage.json"} | set(storage["refs"])
    for rel, digest in storage["refs"].items():
        assert hashlib.sha256((EVAL / rel).read_bytes()).hexdigest() == digest, rel


def test_manifest_unchanged_and_hydrated_set_matches_it():
    snap = json.loads((ROOT / "data/migration_evidence/eval_40pos_v1.1/pre_migration_sha256.json").read_text())
    assert hashlib.sha256((EVAL / "manifest.json").read_bytes()).hexdigest() == snap["files"]["manifest.json"]["sha256"]
    storage = json.loads((EVAL / "storage.json").read_text())
    m = json.loads((EVAL / "manifest.json").read_text())
    assert set(storage["hydrated_files"]) | {"STATS.md"} == set(m["outputs"])


def test_counts_from_refs():
    m = json.loads((EVAL / "manifest.json").read_text())
    inc = [c for c in rows(EVAL / "refs" / "claims.ref.jsonl") if not c["excluded"]]
    assert sum(c["y_true"] for c in inc) == m["counts"]["hallucinated_claims"] >= 40
    assert len(inc) == m["counts"]["claims_scored"]
    assert m["gold_rules_version"] == "1.1" and m["frozen"] is False


def test_disjoint_from_calib_v1_and_one_response_per_source():
    e = rows(EVAL / "refs" / "responses.ref.jsonl")
    c = [json.loads(l) for l in (CALIB / "responses.jsonl").read_text().splitlines()] \
        if (CALIB / "responses.jsonl").exists() else rows(CALIB / "refs" / "responses.ref.jsonl")
    assert {r["split"] for r in e} == {"test"}
    assert not {r["id"] for r in e} & {r["id"] for r in c}
    assert not {r["source_id"] for r in e} & {r["source_id"] for r in c}
    assert len({r["source_id"] for r in e}) == len(e)
    assert {p.stem for p in (EVAL / "refs" / "bundles").glob("*.json")} == {f"ragtruth-{r['id']}" for r in e}
    leak = json.loads((EVAL / "manifest.json").read_text())["leakage"]
    assert all(v == 0 for k, v in leak.items() if k.endswith(("_calib_v1", "_train")))


@needs_hydrated
def test_hydrated_files_are_byte_identical_to_manifest():
    m = json.loads((EVAL / "manifest.json").read_text())
    for rel, digest in m["outputs"].items():
        assert hashlib.sha256((HYD / rel).read_bytes()).hexdigest() == digest, rel


@needs_hydrated
def test_hydrated_claims_match_refs_and_text_offsets():
    refs = rows(EVAL / "refs" / "claims.ref.jsonl")
    full = rows(HYD / "claims.jsonl")
    resp = {r["id"]: r for r in rows(HYD / "responses.jsonl")}
    assert [c["claim_id"] for c in full] == [c["claim_id"] for c in refs]   # identity + order
    for f, r in zip(full, refs):
        assert {k: v for k, v in f.items() if k != "claim_text"} == r       # labels + offsets
        assert resp[f["response_id"]]["response"][f["char_start"]:f["char_end"]] == f["claim_text"]
