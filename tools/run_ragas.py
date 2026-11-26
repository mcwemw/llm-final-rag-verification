"""Run the RAGAS baseline on the example AnswerBundles and measure Det-var.

Usage (from the repo root, venv active):
    python -m tools.run_ragas --limit 1 --runs 1          # smoke test: 1 bundle, 1 run
    python -m tools.run_ragas --runs 3 --out results.json # all 5 bundles, 3 reruns

GROQ_API_KEY is read from the environment; if unset, from a `.env` file in the repo
root or its parent directory (`GROQ_API_KEY=gsk_...`). `.env` is gitignored — never commit it.

Claim-level gold: a claim is "hallucinated" if its [char_start, char_end) overlaps any
RAGTruth human-labelled span. RAGAS predicts "hallucinated" when score < --threshold.
Det-var = variance of claim-level F1 (positive class = hallucinated) across reruns.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from blocks.aligner import align_bundle  # noqa: E402
from blocks.baselines import RagasScorer  # noqa: E402


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


def f1(gold: list[bool], pred: list[bool]) -> float:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="only the first N bundles")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--out", default=None, help="write full results JSON here")
    args = ap.parse_args()

    load_key()
    paths = sorted(glob.glob(str(REPO / "examples" / "answer_bundle_0*.json")))[: args.limit]
    gold_all = json.loads((REPO / "examples" / "answer_bundle_gold.json").read_text(encoding="utf-8"))

    items = []  # (bundle_name, question, claim, cited_docs, gold_hallucinated)
    for p in paths:
        name = Path(p).stem
        bundle = json.loads(Path(p).read_text(encoding="utf-8"))
        spans = [(l["start"], l["end"]) for l in gold_all[name]["labels"]]
        for c in align_bundle(bundle):
            g = any(c.char_start < e and s < c.char_end for s, e in spans)
            items.append((name, bundle["question"], c, bundle["citations"], g))

    scorer = RagasScorer(model=args.model, temperature=args.temperature,
                         price_in_per_mtok=args.price_in, price_out_per_mtok=args.price_out)
    print(f"config: {scorer.config}")
    print(f"{len(paths)} bundles, {len(items)} claims, {args.runs} run(s)\n")

    gold = [it[4] for it in items]
    runs = []
    try:
        for r in range(args.runs):
            t0 = time.perf_counter()
            rows = []
            for name, q, c, docs, g in items:
                scorer.set_question(q)
                res = scorer.score(c, docs)
                rows.append({"bundle": name, "claim_id": res.claim_id, "claim": c.claim_text,
                             "gold_hallucinated": g, "score": res.score,
                             "latency_ms": round(res.latency_ms, 1), "cost_usd": res.cost_usd})
            pred = [row["score"] < args.threshold for row in rows]
            run_f1 = f1(gold, pred)
            runs.append({"run": r + 1, "f1": run_f1, "seconds": round(time.perf_counter() - t0, 1),
                         "claims": rows})
            print(f"run {r + 1}: F1={run_f1:.4f}  ({runs[-1]['seconds']} s)")
    finally:
        scorer.close()

    f1s = [run["f1"] for run in runs]
    per_claim_std = [statistics.pstdev([run["claims"][i]["score"] for run in runs]) for i in range(len(items))]
    n_changed = sum(s > 0 for s in per_claim_std)
    summary = {
        "f1_per_run": f1s,
        "f1_mean": statistics.mean(f1s),
        "det_var_f1": statistics.pvariance(f1s) if len(f1s) > 1 else None,
        "claims_with_score_change": n_changed,
        "n_claims": len(items),
        "total_usage": scorer.total_usage,
        "total_cost_usd": sum(row["cost_usd"] for run in runs for row in run["claims"]),
        "n_nan": scorer.n_nan,
        "n_fallback_question": scorer.n_fallback_question,
        "config": scorer.config,
        "threshold": args.threshold,
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "runs": runs}, indent=2,
                                             ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
