"""Build the FROZEN calibration subset from RAGTruth QA (train split).

Usage (from repo root):
    python3 tools/build_calib_subset.py --data-dir ../RAGTruth/dataset             # oversampled (default)
    python3 tools/build_calib_subset.py --data-dir ../RAGTruth/dataset --natural   # natural hallucination rate

Two layers:
  * FROZEN    : which responses are in the subset (seed + filters + sampling).  -> responses.jsonl
  * DERIVED   : claims + gold, recomputable from the frozen responses with tools/segment.py
                (char offsets into the response text, so a split-rule change only re-derives claims).

Gold per claim (from RAGTruth human span labels, never from an LLM):
  no overlap with any span          -> PASS          (y_true=0)
  overlaps only Baseless-Info spans -> UNSOURCED     (y_true=1)
  overlaps any Conflict span        -> FAIL          (y_true=1)   Conflict wins over Baseless.

Sampling (deterministic):
  pool = task_type QA, split train, quality good, source_id not used in examples/
  sort by int(id) -> random.Random(SEED).shuffle -> round-robin over generator models,
  at most one response per source_id.
  oversampled: phase 1 hallucinated responses until >= --min-halluc-claims hallucinated claims,
               phase 2 faithful responses until >= --target-claims claims.
  --natural  : one round-robin over all responses until >= --target-claims claims.
  Only non-excluded claims count; we stop at response boundaries.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_examples import check_invariants, load_jsonl, to_bundle, validate_schema  # noqa: E402
from segment import MIN_WORDS, RULES_VERSION, segment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261021
CONFLICT = {"Evident Conflict", "Subtle Conflict"}
BASELESS = {"Evident Baseless Info", "Subtle Baseless Info"}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dedupe_labels(labels):
    return sorted({(l["start"], l["end"], l["label_type"]): l for l in labels}.values(),
                  key=lambda l: (l["start"], l["end"], l["label_type"]))


def gold_for(claim, labels):
    s, e = claim["char_start"], claim["char_end"]
    hit = [l for l in labels if l["start"] < e and l["end"] > s]
    types = sorted({l["label_type"] for l in hit})
    covered = set()
    for l in hit:
        covered.update(range(max(s, l["start"]), min(e, l["end"])))
    if any(t in CONFLICT for t in types):
        verdict = "FAIL"
    elif types:
        unknown = set(types) - BASELESS
        assert not unknown, f"unknown label_type {unknown}"
        verdict = "UNSOURCED"
    else:
        verdict = "PASS"
    return {"gold_verdict": verdict, "y_true": int(bool(types)), "label_types": types,
            "overlap_ratio": round(len(covered) / (e - s), 4)}


def claims_for(resp):
    labels = dedupe_labels(resp["labels"])
    out = []
    for k, p in enumerate(segment(resp["response"]), 1):
        c = {"claim_id": f"ragtruth-{resp['id']}-c{k}", "response_id": resp["id"], "source_id": resp["source_id"],
             "model": resp["model"], "claim_text": p.text, "char_start": p.char_start, "char_end": p.char_end,
             "excluded": p.excluded, "exclude_reason": p.exclude_reason}
        c.update(gold_for(c, labels))
        out.append(c)
    return out


def round_robin(by_model):
    """Yield responses alternating over models (sorted model names), each list already shuffled."""
    queues = {m: list(v) for m, v in sorted(by_model.items())}
    while any(queues.values()):
        for m in sorted(queues):
            if queues[m]:
                yield queues[m].pop(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"), help="dir with response.jsonl + source_info.jsonl")
    ap.add_argument("--out", default=str(ROOT / "data" / "frozen" / "calib_v1"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target-claims", type=int, default=150)
    ap.add_argument("--min-halluc-claims", type=int, default=50)
    ap.add_argument("--faithful-share", type=float, default=0.33,
                    help="oversampled mode: fraction of RESPONSES that are fully faithful")
    ap.add_argument("--natural", action="store_true", help="no oversampling: natural hallucination rate")
    args = ap.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out)
    srcs = {s["source_id"]: s for s in load_jsonl(data / "source_info.jsonl") if s["task_type"] == "QA"}
    example_srcs = {g["ragtruth_source_id"]
                    for g in json.loads((ROOT / "examples" / "answer_bundle_gold.json").read_text()).values()}
    pool = [r for r in load_jsonl(data / "response.jsonl")
            if r["source_id"] in srcs and r["split"] == "train" and r["quality"] == "good"
            and r["source_id"] not in example_srcs]
    pool.sort(key=lambda r: int(r["id"]))
    random.Random(args.seed).shuffle(pool)

    # full-split statistics (same segmenter + gold), for the stats table
    def split_stats(resps):
        v = Counter()
        for r in resps:
            for c in claims_for(r):
                if not c["excluded"]:
                    v[c["gold_verdict"]] += 1
        return len(resps), sum(v.values()), v

    full = {sp: split_stats([r for r in load_jsonl(data / "response.jsonl")
                             if r["source_id"] in srcs and r["split"] == sp and r["quality"] == "good"])
            for sp in ("train", "test")}
    _, n_pool_claims, v_pool = split_stats(pool)
    natural_rate = (v_pool["FAIL"] + v_pool["UNSOURCED"]) / n_pool_claims

    picked, used_src = [], set()
    n_claims = n_halluc = 0

    def stream(candidates):
        by_model = defaultdict(list)
        for r in candidates:
            by_model[r["model"]].append(r)
        for r in round_robin(by_model):
            if r["source_id"] in used_src:
                continue
            cl = [c for c in claims_for(r) if not c["excluded"]]
            if cl:
                yield r, cl

    def add(item):
        nonlocal n_claims, n_halluc
        r, cl = item
        picked.append(r)
        used_src.add(r["source_id"])
        n_claims += len(cl)
        n_halluc += sum(c["y_true"] for c in cl)

    if args.natural:
        for item in stream(pool):
            if n_claims >= args.target_claims:
                break
            add(item)
    else:
        # interleave: keep ~faithful_share of RESPONSES faithful, until both targets are met
        hal, fai = stream([r for r in pool if r["labels"]]), stream([r for r in pool if not r["labels"]])
        while not (n_halluc >= args.min_halluc_claims and n_claims >= args.target_claims):
            n_f = sum(1 for r in picked if not r["labels"])
            want_faithful = (n_halluc >= args.min_halluc_claims
                             or n_f < args.faithful_share * (len(picked) + 1))
            if n_claims >= args.target_claims:
                want_faithful = False  # only hallucinated claims still missing
            add(next(fai if want_faithful else hal))

    picked.sort(key=lambda r: int(r["id"]))  # stable file order
    (out / "bundles").mkdir(parents=True, exist_ok=True)
    claims = []
    for r in picked:
        src = srcs[r["source_id"]]
        b = to_bundle(src, r)
        check_invariants(b, src["source_info"]["passages"])
        validate_schema(b)
        (out / "bundles" / f"ragtruth-{r['id']}.json").write_text(
            json.dumps(b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for c in claims_for(r):
            assert r["response"][c["char_start"]:c["char_end"]] == c["claim_text"]
            claims.append(c)

    def write_jsonl(p, rows):
        p.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")

    write_jsonl(out / "responses.jsonl", picked)
    write_jsonl(out / "claims.jsonl", claims)

    # ---- stats ----
    inc = [c for c in claims if not c["excluded"]]
    exc = [c for c in claims if c["excluded"]]
    v = Counter(c["gold_verdict"] for c in inc)
    per_model = Counter(r["model"] for r in picked)
    lost = [c for c in exc if c["y_true"]]
    mode = "natural" if args.natural else (f"oversampled (>= {args.min_halluc_claims} hallucinated claims, "
                                           f"~{args.faithful_share:.0%} of responses faithful)")
    rate = n_halluc / n_claims
    prev = {"QA train (natural)": natural_rate}
    n_t, c_t, v_t = full["test"]
    prev["QA test (natural)"] = (v_t["FAIL"] + v_t["UNSOURCED"]) / c_t
    prev_rows = "".join(f"| {k} | {p:.1%} | {p / rate:.4f} | {(1 - p) / (1 - rate):.4f} |\n" for k, p in prev.items())
    def full_line(sp):
        n_r, n_c, fv = full[sp]
        return (f"{sp:5}  #responses: {n_r:,}   #claims (span->claim): {n_c:,}\n"
                f"       gold: PASS {fv['PASS']:,} / WARN 0* / FAIL {fv['FAIL']:,} / UNSOURCED {fv['UNSOURCED']:,}"
                f"   (hallucinated {(fv['FAIL'] + fv['UNSOURCED']) / n_c:.1%})")

    stats = f"""# Calibration subset calib_v1 — stats

```
RAGTruth QA split (quality=good, segmenter rules v{RULES_VERSION})
{full_line('train')}
{full_line('test')}
  *WARN is a gate output class; gold has only supported / not supported / no source
Calibration subset (frozen, from train, SEED={args.seed}): {len(inc)} claims from {len(picked)} responses
       gold: PASS {v['PASS']} / WARN 0* / FAIL {v['FAIL']} / UNSOURCED {v['UNSOURCED']}
  double-annotation agreement (kappa): not done yet
```

## Calibration subset details

RAGTruth QA, split=train, quality=good, SEED={args.seed}, sampling: {mode}
Segmenter rules v{RULES_VERSION} (tools/segment.py), MIN_WORDS={MIN_WORDS}

| | count |
|---|---|
| responses | {len(picked)} (hallucinated {sum(1 for r in picked if r['labels'])} / faithful {sum(1 for r in picked if not r['labels'])}) |
| claims (calibration, excluded=false) | {len(inc)} |
| gold PASS | {v['PASS']} |
| gold FAIL (Conflict) | {v['FAIL']} |
| gold UNSOURCED (Baseless) | {v['UNSOURCED']} |
| gold WARN | 0 (WARN is a gate output, not a gold class) |
| hallucinated-claim rate, this subset | {rate:.1%} |
| hallucinated-claim rate, natural (whole pool, same segmenter) | {natural_rate:.1%} |
| excluded pieces (meta / short) | {sum(1 for c in exc if c['exclude_reason']=='meta')} / {sum(1 for c in exc if c['exclude_reason']=='short')} |
| excluded pieces overlapping a human span (lost by segmenter) | {len(lost)} |

Per generator model (responses): {', '.join(f'{m} {n}' for m, n in sorted(per_model.items()))}

Gold mapping: no span overlap -> PASS; Baseless only -> UNSOURCED; any Conflict -> FAIL. y_true = 1 if any overlap.
Note: thresholds tuned at the subset rate shift slightly vs the natural rate; final metrics are reported on `test`.

## Prevalence correction

The subset is oversampled ({rate:.1%} hallucinated). To estimate metrics at a natural prevalence pi,
weight each calibration claim by w1 (y_true=1) or w0 (y_true=0) when counting TP/FP/FN/TN:

    w1 = pi / {rate:.4f}        w0 = (1 - pi) / (1 - {rate:.4f})
    precision_pi = w1*TP / (w1*TP + w0*FP)      recall is unchanged (only positives in TP and FN)

| target prevalence pi | value | w1 (y_true=1) | w0 (y_true=0) |
|---|---|---|---|
{prev_rows}
Recall does not depend on prevalence, so a fixed-recall threshold (e.g. recall = 0.8 on hallucinated claims)
transfers from this subset to natural data without correction; max-F1 thresholds should use the weights above.
"""
    (out / "STATS.md").write_text(stats, encoding="utf-8")

    try:
        rt_commit = subprocess.run(["git", "-C", str(data.parent), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        rt_commit = None
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "name": "calib_v1", "seed": args.seed, "sampling": "natural" if args.natural else "oversampled",
        "target_claims": args.target_claims, "faithful_share": None if args.natural else args.faithful_share, "min_halluc_claims": None if args.natural else args.min_halluc_claims,
        "filters": {"task_type": "QA", "split": "train", "quality": "good", "exclude_example_source_ids": True,
                    "one_response_per_source_id": True, "model_round_robin": True},
        "segmenter_rules_version": RULES_VERSION,
        "inputs": {"ragtruth_commit": rt_commit,
                   "response.jsonl": sha256(data / "response.jsonl"),
                   "source_info.jsonl": sha256(data / "source_info.jsonl")},
        "code": {"tools/build_calib_subset.py": sha256(__file__),
                 "tools/segment.py": sha256(Path(__file__).parent / "segment.py")},
        "outputs": {str(p.relative_to(out)): sha256(p) for p in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
