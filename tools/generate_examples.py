"""Convert RAGTruth QA records into AnswerBundle examples (Block 1 -> Block 3 contract).

Usage (from repo root, after fetching data/ per data/README.md):
    python3 tools/generate_examples.py            # writes examples/
    python3 tools/generate_examples.py --n 5 --seed 0

Conventions (to confirm with the Gate owner):
  * One citation per retrieved passage ("passage N:" blocks in source_info.passages).
  * doc_id   = "ragtruth-{source_id}-p{N}"
  * span_*   = character offsets into the ORIGINAL source_info.passages string,
               so that passages[span_start:span_end] == citation.text.
  * Human hallucination labels are NOT part of AnswerBundle (additionalProperties=false);
    they go to a sidecar gold file keyed by example id.
"""
import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "examples"
SCHEMA = ROOT / "schemas" / "answer_bundle.schema.json"

PASSAGE_RE = re.compile(r"passage (\d+):")


def split_passages(passages: str):
    """Return [(N, start, end, text)] with offsets into `passages`."""
    heads = list(PASSAGE_RE.finditer(passages))
    out = []
    for i, m in enumerate(heads):
        start = m.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(passages)
        # trim surrounding whitespace but keep offsets exact
        seg = passages[start:end]
        lstrip = len(seg) - len(seg.lstrip())
        rstrip = len(seg) - len(seg.rstrip())
        s, e = start + lstrip, end - rstrip
        if e > s:
            out.append((int(m.group(1)), s, e, passages[s:e]))
    return out


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_bundle(src, resp):
    passages = src["source_info"]["passages"]
    cits = [
        {"doc_id": f"ragtruth-{src['source_id']}-p{n}", "span_start": s, "span_end": e, "text": t}
        for n, s, e, t in split_passages(passages)
    ]
    # schema v1.1: retrieved = all passages D. In RAGTruth every retrieved passage is also cited.
    retrieved = [{"doc_id": c["doc_id"], "text": c["text"]} for c in cits]
    return {"schema_version": "1.1", "question": src["source_info"]["question"],
            "answer": resp["response"], "citations": cits, "retrieved": retrieved}


def check_invariants(bundle, passages):
    for c in bundle["citations"]:
        assert 0 <= c["span_start"] < c["span_end"] <= len(passages), c["doc_id"]
        assert passages[c["span_start"]:c["span_end"]] == c["text"], c["doc_id"]


def validate_schema(bundle):
    try:
        import jsonschema
    except ImportError:
        print("  (jsonschema not installed; skipped schema validation)")
        return
    schema = json.loads(SCHEMA.read_text())
    # jsonschema<4 has no Draft202012Validator; the keywords used here are draft-7 compatible.
    V = getattr(jsonschema, "Draft202012Validator", jsonschema.Draft7Validator)
    V(schema).validate(bundle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-hallucinated", type=int, default=3)
    ap.add_argument("--split", default="train", help="use train so examples never leak into the test split")
    args = ap.parse_args()

    srcs = {s["source_id"]: s for s in load_jsonl(DATA / "source_info.jsonl") if s["task_type"] == "QA"}
    resps = [
        r for r in load_jsonl(DATA / "response.jsonl")
        if r["source_id"] in srcs and r["split"] == args.split and len(r["response"]) < 900
    ]
    rng = random.Random(args.seed)
    rng.shuffle(resps)

    want_h = args.n_hallucinated
    want_f = args.n - want_h
    picked, used_src, used_model = [], set(), {}
    for r in resps:  # prefer diverse sources / models
        is_h = bool(r["labels"])
        if (is_h and want_h == 0) or (not is_h and want_f == 0):
            continue
        if r["source_id"] in used_src or used_model.get(r["model"], 0) >= 2:
            continue
        picked.append(r)
        used_src.add(r["source_id"])
        used_model[r["model"]] = used_model.get(r["model"], 0) + 1
        if is_h:
            want_h -= 1
        else:
            want_f -= 1
        if want_h == 0 and want_f == 0:
            break

    OUT.mkdir(exist_ok=True)
    gold = {}
    for i, r in enumerate(picked, 1):
        src = srcs[r["source_id"]]
        b = to_bundle(src, r)
        check_invariants(b, src["source_info"]["passages"])
        validate_schema(b)
        name = f"answer_bundle_{i:02d}"
        (OUT / f"{name}.json").write_text(json.dumps(b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        gold[name] = {
            "ragtruth_response_id": r["id"],
            "ragtruth_source_id": r["source_id"],
            "model": r["model"],
            "split": r["split"],
            "hallucinated": bool(r["labels"]),
            # label offsets are into AnswerBundle.answer (== RAGTruth response)
            # RAGTruth sometimes repeats the same span (multiple annotators) -> dedupe
            "labels": sorted(
                {(lab["start"], lab["end"]): {k: lab[k] for k in ("start", "end", "text", "label_type")}
                 for lab in r["labels"]}.values(),
                key=lambda d: d["start"]),
        }
        print(f"OK {name}: response_id={r['id']} model={r['model']} hallucinated={bool(r['labels'])} "
              f"citations={len(b['citations'])}")
    (OUT / "answer_bundle_gold.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
