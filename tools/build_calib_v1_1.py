"""Build calib_v1.1: the frozen calib_v1 responses, with claims and gold re-derived under gold rules v1.1.

Usage (from repo root):
    python3 tools/build_calib_v1_1.py

calib_v1 is not touched. Following calib_v1's own two-layer design:
  * FROZEN layer (which responses): reused from data/frozen/calib_v1/responses.jsonl, after checking
    its sha256 against the calib_v1 manifest. The response set, seed and sampling are unchanged.
  * DERIVED layer (claims + gold): recomputed with tools/gold_rules.py rules v1.1.
Also writes LABEL_CHANGES.md, which lists every claim in the whole RAGTruth QA pool (train + test,
quality=good) whose gold changes from v1.0 to v1.1, and marks the ones in calib_v1 / calib_v1.1.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_calib_subset import sha256  # noqa: E402
from generate_examples import check_invariants, load_jsonl, to_bundle, validate_schema  # noqa: E402
from gold_rules import GOLD_RULES_VERSION, claims_for  # noqa: E402
from segment import MIN_WORDS, RULES_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT / "data" / "frozen" / "calib_v1"
NAME = "calib_v1.1"


def write_jsonl(p, rows):
    p.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def label_changes(responses):
    """[(response, claim_v10, claim_v11)] for every non-excluded claim whose gold differs."""
    out = []
    for r in responses:
        for a, b in zip(claims_for(r, "1.0"), claims_for(r, "1.1")):
            assert a["claim_id"] == b["claim_id"]
            if not a["excluded"] and (a["gold_verdict"], a["y_true"], a["label_types"]) != \
                    (b["gold_verdict"], b["y_true"], b["label_types"]):
                out.append((r, a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "data" / "frozen" / NAME))
    args = ap.parse_args()
    data, out = Path(args.data_dir), Path(args.out)

    pm = json.loads((PARENT / "manifest.json").read_text())
    for name in ("response.jsonl", "source_info.jsonl"):
        assert sha256(data / name) == pm["inputs"][name], f"{name} differs from the calib_v1 snapshot"
    assert sha256(PARENT / "responses.jsonl") == pm["outputs"]["responses.jsonl"], "calib_v1 responses changed"

    srcs = {s["source_id"]: s for s in load_jsonl(data / "source_info.jsonl") if s["task_type"] == "QA"}
    picked = load_jsonl(PARENT / "responses.jsonl")

    (out / "bundles").mkdir(parents=True, exist_ok=True)
    claims = []
    for r in picked:
        src = srcs[r["source_id"]]
        b = to_bundle(src, r)
        check_invariants(b, src["source_info"]["passages"])
        validate_schema(b)
        p = out / "bundles" / f"ragtruth-{r['id']}.json"
        p.write_text(json.dumps(b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        assert sha256(p) == pm["outputs"][f"bundles/{p.name}"], "bundle differs from calib_v1"
        for c in claims_for(r, "1.1"):
            assert r["response"][c["char_start"]:c["char_end"]] == c["claim_text"]
            claims.append(c)
    write_jsonl(out / "responses.jsonl", picked)
    assert sha256(out / "responses.jsonl") == pm["outputs"]["responses.jsonl"]
    write_jsonl(out / "claims.jsonl", claims)

    # ---- label-change report over the whole QA pool ----
    pool = [r for r in load_jsonl(data / "response.jsonl")
            if r["source_id"] in srcs and r["quality"] == "good"]
    changes = label_changes(pool)
    calib_ids = {r["id"] for r in picked}
    n_pos = Counter()
    for r in pool:
        for rules in ("1.0", "1.1"):
            n_pos[(r["split"], rules)] += sum(c["y_true"] for c in claims_for(r, rules) if not c["excluded"])
    rows = []
    for r, a, b in sorted(changes, key=lambda x: (x[0]["split"], int(x[0]["id"]))):
        spans = [l for l in r["labels"] if l["start"] < a["char_end"] and l["end"] > a["char_start"]]
        ov = "; ".join(repr(r["response"][max(a["char_start"], l["start"]):min(a["char_end"], l["end"])])
                       for l in spans)
        where = "**calib_v1 → calib_v1.1**" if r["id"] in calib_ids else ("test pool" if r["split"] == "test"
                                                                            else "train pool")
        rows.append(f"| `{a['claim_id']}` | {r['split']} | {where} | {a['gold_verdict']} (y=1) → "
                    f"{b['gold_verdict']} (y={b['y_true']}) | {ov} | {a['claim_text'][:90]} |\n")
    report = f"""# Gold rules v1.0 → v1.1: label-change report

**Rule change.** A human span overlaps a claim only if the overlapping characters contain at least one
letter or digit (`str.isalnum`). Pure punctuation or whitespace overlap is no longer content overlap.
Segmentation is unchanged (tools/segment.py rules v{RULES_VERSION}); only the gold mapping is versioned.

**Scope.** Every non-excluded claim of every RAGTruth QA response with quality=good (train + test).
The columns compare gold verdict, y_true and label types between the two rules.

| | v1.0 positives | v1.1 positives | changed |
|---|---|---|---|
| train | {n_pos[('train', '1.0')]} | {n_pos[('train', '1.1')]} | {sum(1 for r, _, _ in changes if r['split'] == 'train')} |
| test | {n_pos[('test', '1.0')]} | {n_pos[('test', '1.1')]} | {sum(1 for r, _, _ in changes if r['split'] == 'test')} |

All {len(changes)} changed claims:

| claim_id | split | in | v1.0 → v1.1 | overlapping text | claim |
|---|---|---|---|---|---|
{''.join(rows)}
Response-level labels (`y_true = 1 if labels`) do not depend on this rule, so no response-level
label changes. eval_40pos is built with v1.1 from the start; see its own STATS.md.
"""
    (out / "LABEL_CHANGES.md").write_text(report, encoding="utf-8")

    # ---- stats ----
    inc = [c for c in claims if not c["excluded"]]
    v = Counter(c["gold_verdict"] for c in inc)
    pos = sum(c["y_true"] for c in inc)
    rate = pos / len(inc)
    pool_train = [r for r in pool if r["split"] == "train"]
    nat = {}
    for sp in ("train", "test"):
        cl = [c for r in pool if r["split"] == sp for c in claims_for(r, "1.1") if not c["excluded"]]
        nat[sp] = sum(c["y_true"] for c in cl) / len(cl)
    prev_rows = "".join(f"| QA {sp} (natural, v1.1) | {p:.1%} | {p / rate:.4f} | {(1 - p) / (1 - rate):.4f} |\n"
                        for sp, p in nat.items())
    old_v = Counter(c["gold_verdict"] for c in load_jsonl(PARENT / "claims.jsonl") if not c["excluded"])
    stats = f"""# Calibration subset {NAME} — stats

Same 32 responses as calib_v1 (FROZEN layer reused, sha256-checked); claims and gold re-derived with
gold rules v{GOLD_RULES_VERSION} (tools/gold_rules.py). Segmenter rules v{RULES_VERSION}, MIN_WORDS={MIN_WORDS}.
Seed and sampling are those of calib_v1 (SEED={pm['seed']}, {pm['sampling']}, split=train).

| | calib_v1 (rules v1.0) | {NAME} (rules v{GOLD_RULES_VERSION}) |
|---|---|---|
| responses | {len(picked)} | {len(picked)} |
| claims (excluded=false) | {sum(old_v.values())} | {len(inc)} |
| gold PASS | {old_v['PASS']} | {v['PASS']} |
| gold FAIL (Conflict) | {old_v['FAIL']} | {v['FAIL']} |
| gold UNSOURCED (Baseless) | {old_v['UNSOURCED']} | {v['UNSOURCED']} |
| hallucinated-claim rate | {(old_v['FAIL'] + old_v['UNSOURCED']) / sum(old_v.values()):.1%} | {rate:.1%} |

Changed labels: see LABEL_CHANGES.md.

## Prevalence correction

    w1 = pi / {rate:.4f}        w0 = (1 - pi) / (1 - {rate:.4f})
    precision_pi = w1*TP / (w1*TP + w0*FP)      recall is unchanged

| target prevalence pi | value | w1 (y_true=1) | w0 (y_true=0) |
|---|---|---|---|
{prev_rows}
Thresholds calibrated on calib_v1 must not be reused for {NAME}; recalibrate and bind to this manifest.
"""
    (out / "STATS.md").write_text(stats, encoding="utf-8")

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "name": NAME, "parent": {"name": "calib_v1", "manifest_sha256": sha256(PARENT / "manifest.json"),
                                 "reused": "responses.jsonl (frozen layer) and bundles, sha256-checked"},
        "seed": pm["seed"], "sampling": pm["sampling"], "filters": pm["filters"],
        "segmenter_rules_version": RULES_VERSION, "gold_rules_version": GOLD_RULES_VERSION,
        "counts": {"responses": len(picked), "claims_scored": len(inc), "hallucinated_claims": pos,
                   "PASS": v["PASS"], "FAIL": v["FAIL"], "UNSOURCED": v["UNSOURCED"]},
        "label_changes_vs_parent": [a["claim_id"] for r, a, _ in changes if r["id"] in calib_ids],
        "inputs": {"response.jsonl": sha256(data / "response.jsonl"),
                   "source_info.jsonl": sha256(data / "source_info.jsonl"),
                   "calib_v1/responses.jsonl": sha256(PARENT / "responses.jsonl")},
        "code": {p: sha256(ROOT / p) for p in ("tools/build_calib_v1_1.py", "tools/gold_rules.py",
                                               "tools/build_calib_subset.py", "tools/segment.py",
                                               "tools/generate_examples.py")},
        "outputs": {str(p.relative_to(out)): sha256(p) for p in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(stats)
    print(report)


if __name__ == "__main__":
    main()
