"""Gold rules v1.1 + calib_v1.1 evidence checks (offline; uses only committed files)."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold_rules import claims_for, gold_for_v11  # noqa: E402
from build_calib_subset import gold_for as gold_for_v10  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V10, V11 = ROOT / "data" / "frozen" / "calib_v1", ROOT / "data" / "frozen" / "calib_v1.1"


def rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _manifest_ok(d):
    m = json.loads((d / "manifest.json").read_text())
    on_disk = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert set(m["outputs"]) == on_disk
    for rel, digest in m["outputs"].items():
        assert hashlib.sha256((d / rel).read_bytes()).hexdigest() == digest, rel
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
    _manifest_ok(V10)


def test_v10_path_reproduces_frozen_calib_v1_claims():
    got = [c for r in rows(V10 / "responses.jsonl") for c in claims_for(r, "1.0")]
    text = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in got)
    assert text == (V10 / "claims.jsonl").read_text(encoding="utf-8")


def test_calib_v1_1_evidence():
    m = _manifest_ok(V11)
    assert m["parent"]["manifest_sha256"] == hashlib.sha256((V10 / "manifest.json").read_bytes()).hexdigest()
    assert m["gold_rules_version"] == "1.1" and m["segmenter_rules_version"] == "1.0"
    assert (V11 / "responses.jsonl").read_bytes() == (V10 / "responses.jsonl").read_bytes()
    old = {c["claim_id"]: c for c in rows(V10 / "claims.jsonl")}
    diff = [c["claim_id"] for c in rows(V11 / "claims.jsonl")
            if (c["gold_verdict"], c["y_true"]) != (old[c["claim_id"]]["gold_verdict"], old[c["claim_id"]]["y_true"])]
    assert diff == m["label_changes_vs_parent"] == ["ragtruth-17279-c2"]
