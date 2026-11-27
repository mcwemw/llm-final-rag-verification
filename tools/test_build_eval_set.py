"""Checks on the committed eval_40pos_v1 files (no RAGTruth download needed)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "frozen" / "eval_40pos_v1"
CALIB = ROOT / "data" / "frozen" / "calib_v1"


def rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_manifest_hashes_match_files():
    m = json.loads((EVAL / "manifest.json").read_text())
    on_disk = {str(p.relative_to(EVAL)) for p in EVAL.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert set(m["outputs"]) == on_disk
    for rel, digest in m["outputs"].items():
        assert hashlib.sha256((EVAL / rel).read_bytes()).hexdigest() == digest, rel


def test_at_least_40_positives_and_counts_agree():
    m = json.loads((EVAL / "manifest.json").read_text())
    inc = [c for c in rows(EVAL / "claims.jsonl") if not c["excluded"]]
    assert sum(c["y_true"] for c in inc) == m["counts"]["hallucinated_claims"] >= 40
    assert len(inc) == m["counts"]["claims_scored"]


def test_disjoint_from_calib_v1():
    e, c = rows(EVAL / "responses.jsonl"), rows(CALIB / "responses.jsonl")
    assert {r["split"] for r in e} == {"test"}
    assert not {r["id"] for r in e} & {r["id"] for r in c}
    assert not {r["source_id"] for r in e} & {r["source_id"] for r in c}
    leak = json.loads((EVAL / "manifest.json").read_text())["leakage"]
    assert all(v == 0 for k, v in leak.items() if k.endswith(("_calib_v1", "_train")))


def test_one_bundle_per_response_and_one_response_per_source():
    e = rows(EVAL / "responses.jsonl")
    assert len({r["source_id"] for r in e}) == len(e)
    assert {p.stem for p in (EVAL / "bundles").glob("*.json")} == {f"ragtruth-{r['id']}" for r in e}
