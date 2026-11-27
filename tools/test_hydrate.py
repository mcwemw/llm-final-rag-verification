"""tools/hydrate.py: fail-closed behaviour and "no corpus text in the public repo" checks."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrate as H  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAME = "eval_40pos_v1.1"
SET = ROOT / "data" / "frozen" / NAME
HAVE_CORPUS = all((ROOT / "data" / f).exists() for f in H.CORPUS)
needs_corpus = pytest.mark.skipif(not HAVE_CORPUS, reason="needs local RAGTruth in data/ (see data/README.md)")


def _copy_set(tmp_path):
    frozen = tmp_path / "frozen"
    shutil.copytree(SET, frozen / NAME)
    return frozen


def _hydrate(tmp_path, frozen=None, data_dir=None):
    return H.hydrate(NAME, data_dir=data_dir or ROOT / "data", out_root=tmp_path / "out",
                     repo=tmp_path, frozen=frozen or ROOT / "data" / "frozen")


def _nothing_written(tmp_path):
    out = tmp_path / "out"
    return not out.exists() or not any(p for p in out.iterdir())


# ── fail closed ────────────────────────────────────────────────────────────────
def test_missing_corpus_fails_closed(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(H.HydrationError, match="missing corpus"):
        _hydrate(tmp_path, data_dir=tmp_path / "empty")
    assert _nothing_written(tmp_path)


@needs_corpus
def test_corpus_sha_mismatch_fails_closed(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    shutil.copy(ROOT / "data" / "source_info.jsonl", d)
    raw = (ROOT / "data" / "response.jsonl").read_bytes()
    (d / "response.jsonl").write_bytes(raw[:-2] + b" \n")          # one byte changed
    with pytest.raises(H.HydrationError, match="sha256 differs"):
        _hydrate(tmp_path, data_dir=d)
    assert _nothing_written(tmp_path)


@needs_corpus
def test_tampered_reference_fails_closed(tmp_path):
    frozen = _copy_set(tmp_path)
    p = frozen / NAME / "refs" / "claims.ref.jsonl"
    p.write_text(p.read_text().replace('"char_end": ', '"char_end": 1', 1))
    with pytest.raises(H.HydrationError, match="modified"):
        _hydrate(tmp_path, frozen=frozen)
    assert _nothing_written(tmp_path)


@needs_corpus
def test_offset_out_of_range_fails_closed_even_if_storage_is_updated(tmp_path):
    frozen = _copy_set(tmp_path)
    p = frozen / NAME / "refs" / "claims.ref.jsonl"
    lines = p.read_text().splitlines()
    c = json.loads(lines[0])
    c["char_end"] = 10 ** 6
    lines[0] = json.dumps(c, ensure_ascii=False, sort_keys=True)
    p.write_text("\n".join(lines) + "\n")
    st = json.loads((frozen / NAME / "storage.json").read_text())
    st["refs"]["refs/claims.ref.jsonl"] = H.sha256(p)           # attacker also fixes the ref hash
    (frozen / NAME / "storage.json").write_text(json.dumps(st))
    with pytest.raises(H.HydrationError, match="out of range"):
        _hydrate(tmp_path, frozen=frozen)
    assert _nothing_written(tmp_path)


@needs_corpus
def test_rebuild_not_byte_identical_fails_closed(tmp_path):
    frozen = _copy_set(tmp_path)
    mp = frozen / NAME / "manifest.json"
    m = json.loads(mp.read_text())
    m["outputs"]["claims.jsonl"] = "0" * 64
    mp.write_text(json.dumps(m))
    with pytest.raises(H.HydrationError, match="not byte-identical"):
        _hydrate(tmp_path, frozen=frozen)
    assert _nothing_written(tmp_path) and not list((tmp_path / "out").glob(".*"))   # temp dir removed


def test_refuses_git_tracked_or_unignored_output(tmp_path):
    with pytest.raises(H.HydrationError, match="tracked"):
        H._guard_output(SET, ROOT)                                # a tracked path
    with pytest.raises(H.HydrationError, match="not ignored"):
        H._guard_output(ROOT / "tools" / "hydrated_here", ROOT)   # untracked but not gitignored
    H._guard_output(ROOT / "data" / "hydrated" / NAME, ROOT)       # the intended place: allowed


@needs_corpus
def test_hydrate_succeeds_and_is_byte_identical(tmp_path):
    out = _hydrate(tmp_path)
    m = json.loads((SET / "manifest.json").read_text())
    for rel, digest in m["outputs"].items():
        assert H.sha256(out / rel) == digest


# ── no corpus text in the repo ─────────────────────────────────────────────────
def _tracked_files(path):
    out = subprocess.run(["git", "ls-files", str(path)], cwd=ROOT, capture_output=True, text=True).stdout.split()
    return [ROOT / x for x in out]


def _keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


def test_refs_have_no_text_fields():
    """Structural check: no JSON/JSONL in the set (refs, storage, manifest) carries a text field."""
    files = _tracked_files(SET) or [p for p in SET.rglob("*") if p.is_file()]
    for p in files:
        if p.suffix == ".json":
            docs = [json.loads(p.read_text(encoding="utf-8"))]
        elif p.suffix == ".jsonl":
            docs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            continue
        bad = {k for d in docs for k in _keys(d)} & H.TEXT_KEYS
        assert not bad, f"{p.relative_to(ROOT)} has text field(s) {bad}"


_WORD = re.compile(r"\w+", re.UNICODE)


def _grams(text, n=8):
    w = [t.lower() for t in _WORD.findall(text)]
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


@needs_corpus
def test_no_8gram_of_the_sets_corpus_text_in_tracked_files():
    """Content check: no 8-word window of any response / claim / passage / annotator note of this set
    appears in a tracked file of the set or of the migration evidence."""
    raw = {r["id"]: r for r in H.load_jsonl(ROOT / "data" / "response.jsonl")}
    srcs = {s["source_id"]: s for s in H.load_jsonl(ROOT / "data" / "source_info.jsonl")}
    ids = [json.loads(l)["id"] for l in (SET / "refs" / "responses.ref.jsonl").read_text().splitlines() if l.strip()]
    corpus = set()
    for i in ids:
        r = raw[i]
        corpus |= _grams(r["response"])
        corpus |= _grams(srcs[r["source_id"]]["source_info"]["passages"])
        corpus |= _grams(srcs[r["source_id"]]["source_info"]["question"])
        for l in r["labels"]:
            corpus |= _grams(l.get("text", "")) | _grams(l.get("meta") or "")
    assert len(corpus) > 1000
    files = _tracked_files(SET) + _tracked_files(ROOT / "data" / "migration_evidence" / NAME)
    assert files, "run inside a git checkout"
    for p in files:
        hits = _grams(p.read_text(encoding="utf-8", errors="ignore")) & corpus
        assert not hits, f"{p.relative_to(ROOT)} contains corpus text, e.g. {sorted(hits)[:1]}"
