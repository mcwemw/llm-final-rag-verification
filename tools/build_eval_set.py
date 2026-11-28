"""Build the evaluation set eval_40pos_v1.1 from RAGTruth QA (test split).

Usage (from repo root):
    python3 tools/build_eval_set.py              # -> data/frozen/eval_40pos_v1.1/

Same segmenter and sampling procedure as calib_v1 (tools/build_calib_subset.py) and the same gold
rules as calib_v1.1 (tools/gold_rules.py v1.1), reused by import so the sets cannot drift apart.
Differences:
  * split = test (calib is split = train), new seed; sampling target >= 50 hallucinated claims,
    acceptance floor >= 40 valid positives (asserted);
  * leakage guard: a test source is dropped if its question or ANY of its passages (documents)
    also occurs in a train source. calib_v1 is a subset of train, so this is stricter than
    "no overlap with calib_v1". The overlap with calib_v1 itself is then asserted to be empty
    at response, source, question and document level;
  * per-stratum (generator model x response label) inclusion fractions and weights, so in-sample
    metrics can be separated from population (natural test prevalence) estimates;
  * POSITIVES_REVIEW.md lists every hallucinated claim with the human span it overlaps, for review.

Status: CANDIDATE (manifest "frozen": false) until the team lead signs off.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_calib_subset import dedupe_labels, round_robin, sha256  # noqa: E402
from gold_rules import GOLD_RULES_VERSION, claims_for  # noqa: E402
from generate_examples import check_invariants, load_jsonl, split_passages, to_bundle, validate_schema  # noqa: E402
from segment import MIN_WORDS, RULES_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAME = "eval_40pos_v1.1"
MIN_VALID_POSITIVES = 40
SEED = 20261127  # different from calib_v1 (20261021)
CALIB = ROOT / "data" / "frozen" / "calib_v1.1"  # same responses as calib_v1


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def h(t):
    return hashlib.sha256(norm(t).encode("utf-8")).hexdigest()


def doc_hashes(src):
    return {h(t) for _, _, _, t in split_passages(src["source_info"]["passages"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "data" / "frozen" / NAME))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-halluc-claims", type=int, default=50, help="sampling target")
    ap.add_argument("--target-claims", type=int, default=150)
    ap.add_argument("--faithful-share", type=float, default=0.33,
                    help="fraction of RESPONSES that are fully faithful (same as calib_v1)")
    args = ap.parse_args()

    data, out = Path(args.data_dir), Path(args.out)
    srcs = {s["source_id"]: s for s in load_jsonl(data / "source_info.jsonl") if s["task_type"] == "QA"}
    resps = [r for r in load_jsonl(data / "response.jsonl") if r["source_id"] in srcs]
    example_srcs = {g["ragtruth_source_id"]
                    for g in json.loads((ROOT / "examples" / "answer_bundle_gold.json").read_text()).values()}

    # ---- leakage guard against the whole train split (calib_v1 is drawn from it) ----
    train_src = {r["source_id"] for r in resps if r["split"] == "train"}
    train_docs = set().union(*(doc_hashes(srcs[s]) for s in train_src))
    train_qs = {h(srcs[s]["source_info"]["question"]) for s in train_src}
    test_src = sorted({r["source_id"] for r in resps if r["split"] == "test"}, key=int)
    assert not set(test_src) & train_src, "RAGTruth split shares a source_id"
    dropped = {}
    for s in test_src:
        reasons = []
        if h(srcs[s]["source_info"]["question"]) in train_qs:
            reasons.append("question also in train")
        n = len(doc_hashes(srcs[s]) & train_docs)
        if n:
            reasons.append(f"{n} passage(s) also in train")
        if s in example_srcs:
            reasons.append("used in examples/")
        if reasons:
            dropped[s] = reasons

    pool = [r for r in resps if r["split"] == "test" and r["quality"] == "good" and r["source_id"] not in dropped]
    pool.sort(key=lambda r: int(r["id"]))
    random.Random(args.seed).shuffle(pool)

    # ---- sampling: identical procedure to calib_v1 ----
    picked, used_src = [], set()
    n_claims = n_halluc = 0

    def stream(cands):
        by_model = defaultdict(list)
        for r in cands:
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

    hal, fai = stream([r for r in pool if r["labels"]]), stream([r for r in pool if not r["labels"]])
    while not (n_halluc >= args.min_halluc_claims and n_claims >= args.target_claims):
        n_f = sum(1 for r in picked if not r["labels"])
        want_faithful = n_halluc >= args.min_halluc_claims or n_f < args.faithful_share * (len(picked) + 1)
        if n_claims >= args.target_claims:
            want_faithful = False
        add(next(fai if want_faithful else hal))
    picked.sort(key=lambda r: int(r["id"]))

    # ---- leakage assertions against calib_v1 itself ----
    calib = load_jsonl(CALIB / "responses.jsonl")
    c_ids, c_src = {r["id"] for r in calib}, {r["source_id"] for r in calib}
    c_docs = set().union(*(doc_hashes(srcs[s]) for s in c_src))
    c_qs = {h(srcs[s]["source_info"]["question"]) for s in c_src}
    e_src = {r["source_id"] for r in picked}
    leak = {
        "response_id_overlap_with_calib_v1": len({r["id"] for r in picked} & c_ids),
        "source_id_overlap_with_calib_v1": len(e_src & c_src),
        "question_overlap_with_calib_v1": len({h(srcs[s]["source_info"]["question"]) for s in e_src} & c_qs),
        "document_overlap_with_calib_v1": len(set().union(*(doc_hashes(srcs[s]) for s in e_src)) & c_docs),
        "document_overlap_with_train": len(set().union(*(doc_hashes(srcs[s]) for s in e_src)) & train_docs),
        "split_of_every_response": sorted({r["split"] for r in picked}),
        "one_response_per_source_id": len(e_src) == len(picked),
    }
    assert all(v == 0 for k, v in leak.items() if k.endswith(("_calib_v1", "_train"))), leak
    assert leak["split_of_every_response"] == ["test"] and leak["one_response_per_source_id"], leak

    # ---- write bundles / responses / claims ----
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
        p.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows),
                     encoding="utf-8")

    write_jsonl(out / "responses.jsonl", picked)
    write_jsonl(out / "claims.jsonl", claims)

    # ---- strata: generator model x response label; inclusion fraction and weight ----
    def stratum(r):
        return (r["model"], "hallucinated" if r["labels"] else "faithful")

    # pool restricted to one response per source would be the true frame; responses are the unit drawn
    pool_n = Counter(stratum(r) for r in pool)
    samp_n = Counter(stratum(r) for r in picked)
    strata = [{"model": m, "response_label": lab, "pool_responses": pool_n[(m, lab)],
               "sampled_responses": samp_n[(m, lab)],
               "inclusion_fraction": round(samp_n[(m, lab)] / pool_n[(m, lab)], 4) if pool_n[(m, lab)] else None,
               "weight": round(pool_n[(m, lab)] / samp_n[(m, lab)], 4) if samp_n[(m, lab)] else None}
              for m, lab in sorted(pool_n)]

    inc = [c for c in claims if not c["excluded"]]
    exc = [c for c in claims if c["excluded"]]
    pos = [c for c in inc if c["y_true"]]
    v = Counter(c["gold_verdict"] for c in inc)
    rate = len(pos) / len(inc)
    pool_claims = [c for r in pool for c in claims_for(r) if not c["excluded"]]
    nat = sum(c["y_true"] for c in pool_claims) / len(pool_claims)
    w1, w0 = nat / rate, (1 - nat) / (1 - rate)
    lost = [c for c in exc if c["y_true"]]
    assert len(pos) >= MIN_VALID_POSITIVES, f"only {len(pos)} valid positives (< {MIN_VALID_POSITIVES}): not freezable"
    low_overlap = [c for c in pos if c["overlap_ratio"] < 0.3]
    n_hal_resp = sum(1 for r in picked if r["labels"])
    pool_hal_resp = sum(1 for r in pool if r["labels"])

    strata_rows = "".join(
        f"| {s['model']} | {s['response_label']} | {s['pool_responses']} | {s['sampled_responses']} | "
        f"{s['inclusion_fraction']:.1%} | {s['weight'] if s['weight'] is not None else '–'} |\n" for s in strata)
    dropped_rows = "".join(f"| {s} | {'; '.join(rs)} |\n" for s, rs in sorted(dropped.items(), key=lambda x: int(x[0])))
    per_model_pos = Counter(c["model"] for c in pos)

    stats = f"""# Evaluation set {NAME} — stats (CANDIDATE, not frozen)

RAGTruth QA, **split=test**, quality=good, SEED={args.seed}, segmenter rules v{RULES_VERSION} (MIN_WORDS={MIN_WORDS}),
**gold rules v{GOLD_RULES_VERSION}** (punctuation-only overlap is not a positive).
Sampling is the calib_v1 procedure, imported from `tools/build_calib_subset.py` (one response per source_id,
round-robin over generator models, oversampled: sampling target >= {args.min_halluc_claims} hallucinated claims
(acceptance floor {MIN_VALID_POSITIVES}),
~{args.faithful_share:.0%} of responses fully faithful, stop at response boundaries).

| | count |
|---|---|
| responses | {len(picked)} (hallucinated {n_hal_resp} / faithful {len(picked) - n_hal_resp}) |
| claims (scored, excluded=false) | {len(inc)} |
| **hallucinated claims (y_true=1)** | **{len(pos)}** |
| gold PASS | {v['PASS']} |
| gold FAIL (Conflict) | {v['FAIL']} |
| gold UNSOURCED (Baseless) | {v['UNSOURCED']} |
| hallucinated-claim rate, this set | {rate:.1%} |
| hallucinated-claim rate, natural (whole pool, same segmenter) | {nat:.1%} |
| hallucinated-response rate, this set / pool | {n_hal_resp / len(picked):.1%} / {pool_hal_resp / len(pool):.1%} |
| excluded pieces (meta / short) | {sum(1 for c in exc if c['exclude_reason'] == 'meta')} / {sum(1 for c in exc if c['exclude_reason'] == 'short')} |
| excluded pieces overlapping a human span (lost by segmenter) | {len(lost)} |
| positives with overlap_ratio < 0.3 (see POSITIVES_REVIEW.md) | {len(low_overlap)} |

Hallucinated claims per generator model: {', '.join(f'{m} {n}' for m, n in sorted(per_model_pos.items()))}

## Positive definition

Gold comes only from RAGTruth's human span annotations, never from an LLM (gold rules v{GOLD_RULES_VERSION}, as calib_v1.1):
a claim is hallucinated (y_true=1) if its span overlaps an annotated span on at least one letter or digit;
any Conflict span -> FAIL, Baseless-only -> UNSOURCED, no overlap -> PASS.
Every positive is listed with the overlapping human span and the annotator note in `POSITIVES_REVIEW.md`.

## Leakage checks (all asserted at build time)

| check | result |
|---|---|
| split of every response | {', '.join(leak['split_of_every_response'])} (calib_v1 is train) |
| response_id overlap with calib_v1 / v1.1 | {leak['response_id_overlap_with_calib_v1']} |
| source_id overlap with calib_v1 / v1.1 | {leak['source_id_overlap_with_calib_v1']} |
| question overlap with calib_v1 / v1.1 | {leak['question_overlap_with_calib_v1']} |
| document (passage) overlap with calib_v1 / v1.1 | {leak['document_overlap_with_calib_v1']} |
| document (passage) overlap with the whole train split | {leak['document_overlap_with_train']} |
| one response per source_id | {leak['one_response_per_source_id']} |

Documents are compared per passage after whitespace/case normalisation. Test sources dropped from the pool before sampling:

| source_id | reason |
|---|---|
{dropped_rows}
## Strata and sampling weights

Unit drawn = response. Stratum = generator model x response label. `weight` = pool / sampled
(inverse inclusion fraction) for population estimates at the natural test distribution.

| model | response label | pool | sampled | inclusion | weight |
|---|---|---|---|---|---|
{strata_rows}
**In-sample vs population.** Report in-sample metrics as measured on this set (prevalence {rate:.1%}).
For claim-level estimates at the natural test prevalence ({nat:.1%}), weight each claim by
w1 = {w1:.4f} (y_true=1) or w0 = {w0:.4f} (y_true=0):

    precision_pi = w1*TP / (w1*TP + w0*FP)      recall is unchanged

AUROC and recall do not depend on prevalence; precision, F1 and max-F1 thresholds do.
"""
    (out / "STATS.md").write_text(stats, encoding="utf-8")

    # ---- positives review sheet ----
    by_id = {r["id"]: r for r in picked}
    lines = [f"# {NAME} — hallucinated claims for review ({len(pos)})\n",
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
    (out / "POSITIVES_REVIEW.md").write_text("".join(lines), encoding="utf-8")

    # ---- manifest ----
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "name": NAME, "frozen": False, "status": "candidate: pending team-lead review",
        "seed": args.seed, "sampling": "oversampled",
        "target_claims": args.target_claims, "faithful_share": args.faithful_share,
        "min_halluc_claims": args.min_halluc_claims,
        "filters": {"task_type": "QA", "split": "test", "quality": "good", "exclude_example_source_ids": True,
                    "exclude_sources_sharing_question_or_passage_with_train": True,
                    "one_response_per_source_id": True, "model_round_robin": True},
        "dropped_test_sources": dropped,
        "segmenter_rules_version": RULES_VERSION, "gold_rules_version": GOLD_RULES_VERSION,
        "min_valid_positives": MIN_VALID_POSITIVES,
        "counts": {"responses": len(picked), "claims_scored": len(inc), "hallucinated_claims": len(pos),
                   "PASS": v["PASS"], "FAIL": v["FAIL"], "UNSOURCED": v["UNSOURCED"]},
        "prevalence": {"in_sample": round(rate, 4), "natural_test_pool": round(nat, 4),
                       "w1": round(w1, 4), "w0": round(w0, 4)},
        "strata": strata,
        "leakage": leak,
        "inputs": {"response.jsonl": sha256(data / "response.jsonl"),
                   "source_info.jsonl": sha256(data / "source_info.jsonl"),
                   "calib_v1.1/responses.jsonl": sha256(CALIB / "responses.jsonl")},
        "code": {p: sha256(ROOT / p) for p in ("tools/build_eval_set.py", "tools/gold_rules.py", "tools/build_calib_subset.py",
                                               "tools/segment.py", "tools/generate_examples.py")},
        "outputs": {str(p.relative_to(out)): sha256(p) for p in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
