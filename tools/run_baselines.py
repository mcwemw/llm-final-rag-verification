"""Run the Block 5 baselines over a folder of AnswerBundles; write per-claim scores + summary.

Usage (repo root, venv active):
    # 5 examples, all baselines, 3 reruns
    python -m tools.run_baselines --bundles examples --gold examples/answer_bundle_gold.json \
        --out ../runs/examples
    # calibration subset (bundles + RAGTruth response records holding the human labels)
    python -m tools.run_baselines --bundles ../runs/calib_v1_input/bundles \
        --gold ../runs/calib_v1_input/responses.jsonl --out ../runs/calib_v1
    # smoke test: first 5 claims, 1 run, RAGAS only
    python -m tools.run_baselines ... --scorers ragas --runs 1 --limit-claims 5

Output (--out DIR):
    scores.jsonl  one row per (scorer, run, claim): score, gold y_true, latency, cost.
                  Appended as it goes; rerunning the same command resumes where it stopped.
    summary.json  per scorer: F1/P/R at --threshold per run, Det-var, AUROC, tokens, cost, config.

Gold: a claim is hallucinated (y_true=1) if [char_start, char_end) overlaps any human-labelled
RAGTruth span (labels deduplicated by (start, end)) — the same rule as calib_v1 `y_true`.
Prediction for F1: hallucinated if score < --threshold (scores are "1 = supported").
Claims come from Emma's Block 3/4 (`align_bundle`), so they can differ from calib_v1 claims.jsonl.

GROQ_API_KEY is read from the environment, else from `.env` in the repo root or its parent.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from blocks.aligner import align_bundle  # noqa: E402

# gpt-oss-20b on Groq, USD per 1M tokens (third-party listings, 2026-11; verify at groq.com/pricing)
DEFAULT_PRICE_IN = 0.075
DEFAULT_PRICE_OUT = 0.30
SCORERS = ("presence", "ragas", "selfcheck")


def load_key() -> None:
    if os.environ.get("GROQ_API_KEY"):
        return
    for env in (REPO / ".env", REPO.parent / ".env"):
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                if line.startswith("GROQ_API_KEY="):
                    os.environ["GROQ_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return


def load_gold(path: Path) -> dict[str, list[tuple[int, int]]]:
    """bundle stem -> deduplicated (start, end) hallucination spans."""
    gold: dict[str, list[tuple[int, int]]] = {}
    if path.suffix == ".jsonl":  # RAGTruth response records
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                gold[f"ragtruth-{r['id']}"] = sorted({(l["start"], l["end"]) for l in r["labels"]})
    else:  # {stem: {"labels": [...]}} as in examples/answer_bundle_gold.json
        for stem, g in json.loads(path.read_text(encoding="utf-8")).items():
            gold[stem] = sorted({(l["start"], l["end"]) for l in g["labels"]})
    return gold


def make_scorer(name: str, args):
    from blocks import baselines as b
    if name == "presence":
        return b.PresenceOnlyScorer()
    if name == "ragas":
        return b.RagasScorer(model=args.model, temperature=args.ragas_temperature,
                             price_in_per_mtok=args.price_in, price_out_per_mtok=args.price_out)
    if name == "selfcheck":
        return b.SelfCheckGPTScorer(model=args.model, n_samples=args.sc_samples,
                                    price_in_per_mtok=args.price_in, price_out_per_mtok=args.price_out)
    raise ValueError(name)


def prf(y: list[int], pred: list[int]) -> dict:
    tp = sum(a and b for a, b in zip(y, pred))
    fp = sum((not a) and b for a, b in zip(y, pred))
    fn = sum(a and (not b) for a, b in zip(y, pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"f1": f, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}


def auroc(y: list[int], halluc_score: list[float]) -> float | None:
    """P(random hallucinated claim ranks above a random faithful one); ties count 1/2."""
    pos = [s for s, t in zip(halluc_score, y) if t]
    neg = [s for s, t in zip(halluc_score, y) if not t]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", required=True, help="folder of AnswerBundle *.json files")
    ap.add_argument("--gold", required=True, help="responses.jsonl (RAGTruth) or *_gold.json")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--scorers", default=",".join(SCORERS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit-claims", type=int, default=None, help="first N claims (bundle order)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--ragas-temperature", type=float, default=0.0)
    ap.add_argument("--sc-samples", type=int, default=3)
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN)
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    load_key()
    gold = load_gold(Path(args.gold))
    bundle_paths = sorted(p for p in Path(args.bundles).glob("*.json") if p.stem in gold)
    items = []  # (stem, bundle, claim, y_true)
    for p in bundle_paths:
        bundle = json.loads(p.read_text(encoding="utf-8"))
        for c in align_bundle(bundle):
            y = int(any(c.char_start < e and s < c.char_end for s, e in gold[p.stem]))
            items.append((p.stem, bundle, c, y))
    items = items[: args.limit_claims]
    n_bundles = len({it[0] for it in items})
    print(f"{n_bundles} bundles, {len(items)} claims ({sum(it[3] for it in items)} hallucinated), "
          f"{args.runs} run(s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "scores.jsonl"
    done: dict[tuple, dict] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["method"], r["run"], r["bundle"], r["claim_id"])] = r

    summary = {"bundles": args.bundles, "n_bundles": n_bundles, "n_claims": len(items),
               "n_hallucinated": sum(it[3] for it in items), "threshold": args.threshold,
               "runs": args.runs, "scorers": {}}
    with rows_path.open("a", encoding="utf-8") as fout:
        for name in args.scorers.split(","):
            scorer = make_scorer(name, args)
            method = scorer.name
            per_run = []
            try:
                for run in range(1, args.runs + 1):
                    if hasattr(scorer, "reset_samples"):
                        scorer.reset_samples()  # fresh SelfCheckGPT samples every rerun
                    t0, rows, n_new = time.perf_counter(), [], 0
                    for i, (stem, bundle, c, y) in enumerate(items):
                        cid = f"c{c.char_start}"
                        key = (method, run, stem, cid)
                        if key not in done:
                            if hasattr(scorer, "set_question"):
                                scorer.set_question(bundle["question"])
                            res = scorer.score(c, bundle["citations"])
                            done[key] = {"method": method, "run": run, "bundle": stem,
                                         "claim_id": res.claim_id, "claim": c.claim_text,
                                         "alignment": c.alignment, "y_true": y, "score": res.score,
                                         "latency_ms": round(res.latency_ms, 1), "cost_usd": res.cost_usd}
                            fout.write(json.dumps(done[key], ensure_ascii=False) + "\n")
                            fout.flush()
                            n_new += 1
                        rows.append(done[key])
                        if n_new and (i + 1) % 20 == 0:
                            print(f"  {method} run {run}: {i + 1}/{len(items)}", flush=True)
                    ys = [r["y_true"] for r in rows]
                    m = prf(ys, [int(r["score"] < args.threshold) for r in rows])
                    m["auroc"] = auroc(ys, [1 - r["score"] for r in rows])
                    m["cost_usd"] = sum(r["cost_usd"] for r in rows)
                    m["seconds_this_session"] = round(time.perf_counter() - t0, 1)
                    per_run.append(m)
                    print(f"{method} run {run}: F1={m['f1']:.4f} P={m['precision']:.3f} "
                          f"R={m['recall']:.3f} AUROC={m['auroc']} cost=${m['cost_usd']:.4f}", flush=True)
            finally:
                if hasattr(scorer, "close"):
                    scorer.close()

            f1s = [m["f1"] for m in per_run]
            scores = [[done[(method, r, it[0], f"c{it[2].char_start}")]["score"] for it in items]
                      for r in range(1, args.runs + 1)]
            changed = sum(len(set(col)) > 1 for col in zip(*scores)) if args.runs > 1 else 0
            summary["scorers"][method] = {
                "per_run": per_run,
                "f1_mean": statistics.mean(f1s),
                "det_var_f1_population": statistics.pvariance(f1s) if len(f1s) > 1 else None,
                "det_var_f1_sample": statistics.variance(f1s) if len(f1s) > 1 else None,
                "claims_with_score_change": changed,
                "total_cost_usd": sum(m["cost_usd"] for m in per_run),
                "cost_per_claim_usd": sum(m["cost_usd"] for m in per_run) / (len(items) * args.runs),
                "total_usage_this_session": getattr(scorer, "total_usage", None),
                "n_nan": getattr(scorer, "n_nan", None),
                "n_empty_samples": getattr(scorer, "n_empty_samples", None),
                "n_fallback_question": getattr(scorer, "n_fallback_question", None),
                "config": getattr(scorer, "config", {"method": method}),
            }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("f1_mean", "det_var_f1_population",
                                                "claims_with_score_change", "total_cost_usd")}
                      for k, v in summary["scorers"].items()}, indent=2))
    print(f"wrote {out}/scores.jsonl, {out}/summary.json")


if __name__ == "__main__":
    main()
