"""Reference-only storage for frozen sets + runtime hydration from a local RAGTruth copy.

The public repo keeps no corpus text (RAGTruth QA passages come from MS MARCO: "non-commercial
research purposes only ... without extending any license"). A frozen set keeps:
    manifest.json   unchanged: the sha256 of every FULL file, i.e. the acceptance target
    STATS.md        our own statistics (no corpus text)
    storage.json    which files are stored as references, the sha256 of each reference file
    refs/           ids, offsets and labels only (no question / answer / text / claim_text / meta)

Hydrate (each user, at run time):
    python3 tools/hydrate.py eval_40pos_v1.1          # -> data/hydrated/eval_40pos_v1.1/ (gitignored)

It FAILS CLOSED and writes nothing when:
  * data/response.jsonl or data/source_info.jsonl is missing, or its sha256 differs from the manifest;
  * a reference file's sha256 differs from storage.json;
  * an id is not in the corpus, an offset is out of range, or a reference disagrees with the corpus;
  * any rebuilt file's sha256 differs from manifest.json (byte-identical or nothing);
  * the output path is tracked by git or not ignored by .gitignore.
Output is built in a temporary directory and moved into place only after every check passes.

Maintainers convert a full set to references once with:  python3 tools/hydrate.py --make-refs <set>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_calib_subset import dedupe_labels  # noqa: E402
from generate_examples import load_jsonl, to_bundle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "data" / "frozen"
DATA = ROOT / "data"
HYDRATED = ROOT / "data" / "hydrated"
REF_FORMAT = "ragtruth-ref-v1"
CORPUS = ("response.jsonl", "source_info.jsonl")
TEXT_KEYS = {"question", "answer", "text", "claim_text", "response", "meta"}  # never in refs
LABEL_KEEP = ("start", "end", "label_type", "due_to_null", "implicit_true")
RESP_KEEP = ("id", "source_id", "model", "split", "quality", "temperature")


class HydrationError(RuntimeError):
    """Any failed check. Nothing is written when this is raised."""


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _dumps_line(x: dict) -> str:  # identical to the builders' write_jsonl
    return json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n"


def _dump_bundle(b: dict) -> str:  # identical to the builders' bundle writer
    return json.dumps(b, ensure_ascii=False, indent=2) + "\n"


# ── refs ──────────────────────────────────────────────────────────────────────
def response_ref(r: dict) -> dict:
    ref = {k: r[k] for k in RESP_KEEP if k in r}
    ref["labels"] = [{k: l[k] for k in LABEL_KEEP if k in l} for l in r["labels"]]
    return ref


def claim_ref(c: dict) -> dict:
    return {k: v for k, v in c.items() if k != "claim_text"}


def bundle_ref(b: dict, resp: dict) -> dict:
    return {"ref_format": REF_FORMAT, "schema_version": b.get("schema_version"),
            "response_id": resp["id"], "source_id": resp["source_id"],
            "citations": [{k: c[k] for k in ("doc_id", "span_start", "span_end")} for c in b["citations"]],
            "retrieved": [{"doc_id": d["doc_id"]} for d in b.get("retrieved", [])]}


def make_refs(name: str) -> dict:
    """Write refs/ + storage.json for a full set. Returns storage. Does not delete anything."""
    set_dir = FROZEN / name
    man = json.loads((set_dir / "manifest.json").read_text())
    resps = load_jsonl(set_dir / "responses.jsonl")
    claims = load_jsonl(set_dir / "claims.jsonl")
    refs = set_dir / "refs"
    (refs / "bundles").mkdir(parents=True, exist_ok=True)
    (refs / "responses.ref.jsonl").write_text("".join(_dumps_line(response_ref(r)) for r in resps), encoding="utf-8")
    (refs / "claims.ref.jsonl").write_text("".join(_dumps_line(claim_ref(c)) for c in claims), encoding="utf-8")
    for r in resps:
        b = json.loads((set_dir / "bundles" / f"ragtruth-{r['id']}.json").read_text(encoding="utf-8"))
        (refs / "bundles" / f"ragtruth-{r['id']}.json").write_text(
            json.dumps(bundle_ref(b, r), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hydrated = sorted(k for k in man["outputs"] if k != "STATS.md")
    storage = {
        "name": name, "storage": "references", "ref_format": REF_FORMAT,
        "why": "no corpus text in the public repo (RAGTruth / MS MARCO terms); rebuild locally",
        "hydrate": f"python3 tools/hydrate.py {name}",
        "kept_in_repo": ["manifest.json", "STATS.md", "storage.json", "refs/"],
        "hydrated_files": hydrated,
        "acceptance": "every hydrated file's sha256 == manifest.json outputs (byte-identical)",
        "refs": {str(p.relative_to(set_dir)): sha256(p) for p in sorted(refs.rglob("*")) if p.is_file()},
    }
    (set_dir / "storage.json").write_text(json.dumps(storage, indent=2) + "\n", encoding="utf-8")
    return storage


# ── hydrate ───────────────────────────────────────────────────────────────────
def _positives_review(name: str, picked: list[dict], claims: list[dict]) -> str:
    """Byte-for-byte the review sheet tools/build_eval_set.py writes (that file is not modified,
    so its sha in the manifest stays valid); the output sha check proves the equivalence."""
    pos = [c for c in claims if not c["excluded"] and c["y_true"]]
    by_id = {r["id"]: r for r in picked}
    lines = [f"# {name} — hallucinated claims for review ({len(pos)})\n",
             "Each row: claim (from the response) and the RAGTruth human span(s) it overlaps. "
             "`overlap` = fraction of the claim's characters covered by a span. "
             "Rows with overlap < 0.3 are marked ⚠ (claim only touches the span edge).\n"]
    for i, c in enumerate(pos, 1):
        r = by_id[c["response_id"]]
        spans = [l for l in dedupe_labels(r["labels"])
                 if any(r["response"][i].isalnum() for i in range(max(l["start"], c["char_start"]), min(l["end"], c["char_end"])))]
        warn = " ⚠" if c["overlap_ratio"] < 0.3 else ""
        lines.append(f"\n## {i}. `{c['claim_id']}` — {c['gold_verdict']}, overlap {c['overlap_ratio']}{warn}\n")
        lines.append(f"- model: {c['model']}\n- claim: {c['claim_text']}\n")
        for l in spans:
            note = " / ".join(x.strip() for x in (l.get("meta") or "").splitlines() if x.strip())
            lines.append(f"- span ({l['label_type']}): {l['text']}\n  - annotator note: {note}\n")
    return "".join(lines)


def _guard_output(out: Path, repo: Path) -> None:
    rel = out.resolve().relative_to(repo.resolve()) if out.resolve().is_relative_to(repo.resolve()) else None
    if rel is None:
        return  # outside the repo: cannot be committed
    tracked = subprocess.run(["git", "ls-files", "--", str(rel)], cwd=repo, capture_output=True, text=True).stdout
    if tracked.strip():
        raise HydrationError(f"refusing to write into git-tracked path {rel}")
    probe = subprocess.run(["git", "check-ignore", "-q", "--no-index", str(rel / "probe.json")], cwd=repo)
    if probe.returncode != 0:
        raise HydrationError(f"refusing to write into {rel}: not ignored by .gitignore")


def hydrate(name: str, data_dir: Path = DATA, out_root: Path = HYDRATED, repo: Path = ROOT,
            frozen: Path = FROZEN) -> Path:
    set_dir = frozen / name
    man = json.loads((set_dir / "manifest.json").read_text())
    storage = json.loads((set_dir / "storage.json").read_text())
    out = out_root / name
    _guard_output(out, repo)

    for fname in CORPUS:  # 1. local corpus present and identical to the snapshot the set was built from
        p = data_dir / fname
        if not p.is_file():
            raise HydrationError(f"missing corpus file {p} (fetch RAGTruth, see data/README.md)")
        if sha256(p) != man["inputs"][fname]:
            raise HydrationError(f"{p} sha256 differs from manifest inputs (different RAGTruth snapshot)")
    for rel, digest in storage["refs"].items():  # 2. references untouched
        p = set_dir / rel
        if not p.is_file() or sha256(p) != digest:
            raise HydrationError(f"reference file {rel} missing or modified")

    raw = {r["id"]: r for r in load_jsonl(data_dir / "response.jsonl")}
    srcs = {s["source_id"]: s for s in load_jsonl(data_dir / "source_info.jsonl")}
    refs = set_dir / "refs"
    files: dict[str, str] = {}

    picked = []  # 3. responses: identity + labels + offsets checked against the corpus
    for ref in load_jsonl(refs / "responses.ref.jsonl"):
        r = raw.get(ref["id"])
        if r is None:
            raise HydrationError(f"response {ref['id']} not in corpus")
        if response_ref(r) != ref:
            raise HydrationError(f"response {ref['id']}: reference disagrees with corpus (ids/labels/offsets)")
        for l in r["labels"]:
            if not 0 <= l["start"] <= l["end"] <= len(r["response"]):
                raise HydrationError(f"response {ref['id']}: label offset out of range")
        picked.append(r)
    files["responses.jsonl"] = "".join(_dumps_line(r) for r in picked)

    by_id = {r["id"]: r for r in picked}
    claims = []  # 4. claims: text = response[char_start:char_end]
    for ref in load_jsonl(refs / "claims.ref.jsonl"):
        r = by_id.get(ref["response_id"])
        if r is None:
            raise HydrationError(f"claim {ref['claim_id']}: response not in set")
        s, e = ref["char_start"], ref["char_end"]
        if not 0 <= s < e <= len(r["response"]):
            raise HydrationError(f"claim {ref['claim_id']}: offset [{s},{e}) out of range")
        claims.append({**ref, "claim_text": r["response"][s:e]})
    files["claims.jsonl"] = "".join(_dumps_line(c) for c in claims)

    for r in picked:  # 5. bundles: rebuilt from the corpus, checked against the reference
        ref = json.loads((refs / "bundles" / f"ragtruth-{r['id']}.json").read_text(encoding="utf-8"))
        src = srcs.get(r["source_id"])
        if src is None:
            raise HydrationError(f"source {r['source_id']} not in corpus")
        b = to_bundle(src, r)
        if bundle_ref(b, r) != ref:
            raise HydrationError(f"bundle ragtruth-{r['id']}: reference disagrees with rebuilt bundle")
        passages = src["source_info"]["passages"]
        for c in b["citations"]:
            if not 0 <= c["span_start"] < c["span_end"] <= len(passages):
                raise HydrationError(f"bundle ragtruth-{r['id']}: span out of range")
        files[f"bundles/ragtruth-{r['id']}.json"] = _dump_bundle(b)

    if "POSITIVES_REVIEW.md" in man["outputs"]:
        files["POSITIVES_REVIEW.md"] = _positives_review(name, picked, claims)

    expected = set(storage["hydrated_files"])
    if set(files) != expected:
        raise HydrationError(f"rebuilt file set differs from storage.json: {sorted(set(files) ^ expected)}")

    out_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=out_root))
    try:
        for rel, content in files.items():  # 6. byte-identical or nothing
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            if sha256(p) != man["outputs"][rel]:
                raise HydrationError(f"rebuilt {rel} is not byte-identical to the manifest")
        shutil.copy2(set_dir / "manifest.json", tmp / "manifest.json")
        shutil.copy2(set_dir / "STATS.md", tmp / "STATS.md")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", help="frozen set name, e.g. eval_40pos_v1.1")
    ap.add_argument("--make-refs", action="store_true", help="maintainers: write refs/ + storage.json")
    args = ap.parse_args()
    if args.make_refs:
        s = make_refs(args.set)
        print(f"wrote {len(s['refs'])} reference files + storage.json for {args.set}")
        return
    try:
        out = hydrate(args.set)
    except HydrationError as e:
        sys.exit(f"HYDRATION FAILED (nothing written): {e}")
    print(f"hydrated {args.set} -> {out} (every file byte-identical to manifest.json)")


if __name__ == "__main__":
    main()
