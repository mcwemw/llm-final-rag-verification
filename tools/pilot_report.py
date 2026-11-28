"""Judge-pilot report for one or more run folders written by tools/run_baselines.py.

    python -m tools.pilot_report runs/pilot_nemotron_cited_span [more run dirs] --out report.json

Rules (fixed before looking at Phase 1 results):
  * The last row per (method, run, claim_id) in scores.jsonl is the result for that cell.
  * Legacy logs (before 2026-11-28) wrote an invalid judge output as status "judge_nan" with
    score 0.0. Such rows are read as status "invalid_output", score None, and listed under
    "relabelled" so the change is traceable. The log files themselves are never modified.
  * Quality metrics use only rows with status "ok". Excluded and counted separately:
    skipped_no_source (the claim has no cited source to judge), invalid_output (the judge gave
    no statements), error (API failure). None of them is ever scored as 0.
  * Prediction: unfaithful if score < threshold (default 0.5, the runner default).
    False Entailment = gold-unfaithful claims predicted faithful / gold-unfaithful claims.
    False unfaithful = gold-faithful claims predicted unfaithful / gold-faithful claims.
  * Determinism: among claims with an "ok" row in every run, how many have identical scores.
    Claims with any non-ok run are reported separately, never counted as stable.
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def auroc(y, h):
    pos = [s for t, s in zip(y, h) if t == 1]
    neg = [s for t, s in zip(y, h) if t == 0]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def load_cells(run_dir: Path):
    cells, relabelled = {}, []
    for line in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cells[(r["method"], r["run"], r["claim_id"])] = r
    for k, r in cells.items():
        if r["status"] == "judge_nan":
            relabelled.append({"run": k[1], "claim_id": k[2], "logged_score": r["score"]})
            cells[k] = {**r, "status": "invalid_output", "score": None,
                        "error": "invalid_output: judge returned no statements (legacy judge_nan)"}
    return cells, relabelled


def report(run_dir: Path, threshold: float = 0.5) -> dict:
    cells, relabelled = load_cells(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    runs = sorted({k[1] for k in cells})
    claims = sorted({k[2] for k in cells})
    per_run = []
    for run in runs:
        rows = [cells[k] for k in cells if k[1] == run]
        ok = [r for r in rows if r["status"] == "ok"]
        y = [r["y_true"] for r in ok]
        pred = [int(r["score"] < threshold) for r in ok]
        npos, nneg = sum(y), len(y) - sum(y)
        fe = sum(t == 1 and p == 0 for t, p in zip(y, pred))
        fu = sum(t == 0 and p == 1 for t, p in zip(y, pred))
        per_run.append({
            "run": run, "n_rows": len(rows), "n_scored": len(ok), "n_positive": npos,
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "auroc": auroc(y, [1 - r["score"] for r in ok]),
            "gold_agreement": sum(t == p for t, p in zip(y, pred)) / len(ok) if ok else None,
            "false_entailment": {"n": fe, "of": npos, "rate": fe / npos if npos else None},
            "false_unfaithful": {"n": fu, "of": nneg, "rate": fu / nneg if nneg else None},
            "invalid_output_rate": (sum(r["status"] == "invalid_output" for r in rows)
                                    / sum(r["status"] != "skipped_no_source" for r in rows)),
        })
    det = None
    if len(runs) > 1:
        all_ok = [c for c in claims if all(cells.get((m, r, c), {}).get("status") == "ok"
                                           for m in {k[0] for k in cells} for r in runs)]
        same = [c for c in all_ok if len({cells[(m, r, c)]["score"] for m in {k[0] for k in cells}
                                          for r in runs}) == 1]
        not_all_ok = {c: [cells.get((m, r, c), {}).get("status") for m in {k[0] for k in cells}
                          for r in runs] for c in claims if c not in all_ok}
        det = {"claims_ok_in_every_run": len(all_ok), "identical_scores": len(same),
               "claims_not_ok_in_every_run": {c: s for c, s in not_all_ok.items()
                                              if set(s) != {"skipped_no_source"}},
               "n_claims_skipped_no_source": sum(set(s) == {"skipped_no_source"}
                                                 for s in not_all_ok.values())}
    rows = list(cells.values())
    calls_log = run_dir / "judge_calls.jsonl"
    calls = [json.loads(l) for l in calls_log.read_text(encoding="utf-8").splitlines()
             if l.strip()] if calls_log.exists() else []
    call_lat = [c["latency_ms"] for a in calls for c in a["calls"] if c.get("latency_ms") is not None]
    http = Counter(h["status"] for a in calls for h in a["http"])
    claim_lat = [r["latency_ms"] for r in rows if r["status"] in ("ok", "invalid_output")]
    return {
        "run_dir": str(run_dir), "run_id": manifest["run_id"],
        "dataset": {k: manifest["dataset"].get(k) for k in ("version", "sha256", "n_claims",
                                                              "gold_rules_version")},
        "model": manifest["scorers"].get("ragas", {}).get("config", {}).get("model"),
        "context": manifest["scorers"].get("ragas", {}).get("config", {}).get("context"),
        "code_commit": manifest["code"].get("commit"),
        "threshold": threshold, "relabelled_legacy_judge_nan": relabelled,
        "per_run": per_run, "determinism": det,
        "operations": {
            "claim_evaluations": len(rows),
            "llm_calls": sum(len(a["calls"]) for a in calls),
            "http_requests": sum(http.values()),
            "http_status_counts": {str(k): v for k, v in sorted(http.items())},
            "finish_reasons": dict(Counter(c.get("finish_reason") for a in calls for c in a["calls"])),
            "models_returned": sorted({c.get("model_returned") for a in calls for c in a["calls"]} - {None}),
            "tokens": {k: sum(r.get(k, 0) for r in rows)
                       for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")},
            "call_latency_s": {"mean": statistics.mean(call_lat) / 1000, "p95": pct(call_lat, .95) / 1000}
            if call_lat else None,
            "claim_latency_s": {"mean": statistics.mean(claim_lat) / 1000, "p95": pct(claim_lat, .95) / 1000}
            if claim_lat else None,
            "cost_usd": sum(r.get("cost_usd") or 0 for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()
    reps = [report(Path(d), args.threshold) for d in args.run_dirs]
    text = json.dumps(reps, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    for r in reps:
        print(f"== {r['run_dir']}  model={r['model']} context={r['context']}")
        for m in r["per_run"]:
            print(f"  run{m['run']} scored={m['n_scored']} pos={m['n_positive']} AUROC={m['auroc']:.3f} "
                  f"agree={m['gold_agreement']:.3f} FE={m['false_entailment']['n']}/{m['false_entailment']['of']}"
                  f"={m['false_entailment']['rate']:.3f} FU={m['false_unfaithful']['n']}/"
                  f"{m['false_unfaithful']['of']}={m['false_unfaithful']['rate']:.3f} "
                  f"invalid={m['invalid_output_rate']:.4f} {m['status_counts']}")
        if r["determinism"]:
            d = r["determinism"]
            print(f"  identical {d['identical_scores']}/{d['claims_ok_in_every_run']}; "
                  f"not ok in every run: {d['claims_not_ok_in_every_run']}")
        print(f"  relabelled: {r['relabelled_legacy_judge_nan']}")
        print(f"  ops: {json.dumps(r['operations'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
