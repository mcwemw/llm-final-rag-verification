"""Run the Block 5 baselines over a folder of AnswerBundles; write per-claim scores + run metadata.

Usage (repo root, venv active):
    # pilot: 3 calib_v1 bundles, all baselines, 3 reruns, Fireworks, hard budget $0.20
    python -m tools.run_baselines --bundles ../runs/calib_v1_input/bundles \
        --gold ../runs/calib_v1_input/responses.jsonl --dataset-version calib_v1 \
        --provider fireworks --limit-bundles 3 --max-cost-usd 0.2 --out ../runs/pilot
    # then convert to score-export-v1 for the harness:
    python -m tools.export_scores --run-dir ../runs/pilot

Output (--out DIR):
    manifest.json  run id, git sha, packages, dataset (bundle ids + sha256), and the exact
                   config of every scorer. Resuming with a different config / dataset / code
                   is refused: one run directory = one experimental condition.
    scores.jsonl   one row per (scorer, rerun, claim): score, status, error, y_true, latency,
                   LLM tokens / calls / HTTP requests / retries, NLI pairs / ms, cost.
                   Appended as it goes; rerunning the same command resumes, and retries rows
                   whose status was "error". The last row for a key wins.
    summary.json   per scorer: F1/P/R at --threshold per rerun, AUROC, Det-var, coverage,
                   determinism_trusted, LLM usage, NLI usage, estimated vs actual cost.

Budget: --max-cost-usd is a hard cap on the ACTUAL cost recorded in this run directory (all
reruns and resumes, retries and failed calls included). The run stops cleanly when reached.

Gold: a claim is hallucinated (y_true=1) if [char_start, char_end) overlaps any human-labelled
RAGTruth span (labels deduplicated by (start, end)) — the same rule as calib_v1 `y_true`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from blocks.aligner import align_bundle  # noqa: E402

SCORERS = ("presence", "ragas", "selfcheck")

# USD per 1M tokens (input, output), with where the number comes from.
PRICES = {
    ("groq", "openai/gpt-oss-20b"): (0.075, 0.30, "third-party listings, 2026-11 (unverified)"),
    ("fireworks", "accounts/fireworks/models/glm-5p3-flash"):
        (0.15, 0.50, "app.fireworks.ai/models, uncached, read 2026-11-27"),
}
# Pre-run cost estimate inputs, measured in the 2026-11-27 model comparison (6 claims / 1 response):
#   RAGAS per claim, SelfCheckGPT sampling per response (3 samples).
EST_TOKENS = {
    ("ragas", "accounts/fireworks/models/glm-5p3-flash"): (1712, 1023),
    ("selfcheck", "accounts/fireworks/models/glm-5p3-flash"): (1512, 511),
    ("ragas", "openai/gpt-oss-20b"): (1877, 1043),
    ("selfcheck", "openai/gpt-oss-20b"): (1059, 307),
}
STOP_ERRORS = ("RateLimitError", "AuthenticationError", "PermissionDeniedError",
               "APIConnectionError", "NotFoundError", "InternalServerError")


class BudgetExceeded(Exception):
    pass


def load_env() -> None:
    """Fill API keys from .env (repo root or its parent) without overriding the environment."""
    for env in (REPO / ".env", REPO.parent / ".env"):
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip().removeprefix("export ")
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    if k.endswith("_API_KEY"):
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


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


def git_info() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()
    tracked_dirty = run("git", "status", "--porcelain", "--untracked-files=no")
    return {"git_sha": run("git", "rev-parse", "HEAD"), "git_dirty": bool(tracked_dirty)}


def packages() -> dict:
    out = {"python": platform.python_version()}
    for p in ("ragas", "openai", "instructor", "transformers", "torch", "httpx"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def make_scorer(name: str, args, price_in, price_out):
    from blocks import baselines as b
    if name == "presence":
        return b.PresenceOnlyScorer()
    common = dict(provider=args.provider, model=args.model, seed=args.seed,
                  price_in_per_mtok=price_in, price_out_per_mtok=price_out)
    if name == "ragas":
        return b.RagasScorer(temperature=args.ragas_temperature, **common)
    if name == "selfcheck":
        return b.SelfCheckGPTScorer(n_samples=args.sc_samples, **common)
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


def scorer_stub_config(name: str, args) -> dict:
    """Config that identifies the experimental condition, known before building the scorer."""
    return {"scorer": name, "provider": args.provider if name != "presence" else None,
            "model": args.model if name != "presence" else None, "seed": args.seed,
            "ragas_temperature": args.ragas_temperature if name == "ragas" else None,
            "sc_samples": args.sc_samples if name == "selfcheck" else None,
            "threshold": args.threshold}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", required=True, help="folder of AnswerBundle *.json files")
    ap.add_argument("--gold", required=True, help="responses.jsonl (RAGTruth) or *_gold.json")
    ap.add_argument("--dataset-version", required=True, help="e.g. calib_v1 | ragtruth-qa-test-strat100")
    ap.add_argument("--out", required=True, help="output folder (one folder = one experimental condition)")
    ap.add_argument("--scorers", default=",".join(SCORERS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit-bundles", type=int, default=None, help="first N bundles (sorted by id)")
    ap.add_argument("--bundle-ids", default=None, help="file with one bundle id per line (overrides order)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--provider", default="fireworks", choices=["fireworks", "groq"])
    ap.add_argument("--model", default=None, help="default: the provider's default model")
    ap.add_argument("--seed", type=int, default=None,
                    help="off by default: a fixed sampling seed would suppress the run-to-run "
                         "variance that Det-var is meant to measure")
    ap.add_argument("--ragas-temperature", type=float, default=0.0)
    ap.add_argument("--sc-samples", type=int, default=3)
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--max-cost-usd", type=float, default=2.0, help="hard cap on actual cost in --out")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from blocks.baselines import PROVIDERS
    args.model = args.model or PROVIDERS[args.provider]["model"]
    price = PRICES.get((args.provider, args.model))
    price_in = args.price_in if args.price_in is not None else (price[0] if price else None)
    price_out = args.price_out if args.price_out is not None else (price[1] if price else None)
    price_source = "command line" if args.price_in is not None else (price[2] if price else None)
    scorer_names = args.scorers.split(",")
    if any(n != "presence" for n in scorer_names) and (price_in is None or price_out is None):
        sys.exit(f"No known price for {args.provider}/{args.model}: pass --price-in and --price-out.")

    load_env()
    gold = load_gold(Path(args.gold))
    paths = {p.stem: p for p in Path(args.bundles).glob("*.json") if p.stem in gold}
    if args.bundle_ids:
        ids = [l.strip() for l in Path(args.bundle_ids).read_text().splitlines() if l.strip()]
        missing = [i for i in ids if i not in paths]
        if missing:
            sys.exit(f"bundle ids not found: {missing[:5]}")
    else:
        ids = sorted(paths)
    ids = ids[: args.limit_bundles]
    sha = hashlib.sha256()
    items = []  # (stem, bundle, claim, y_true)
    for stem in ids:
        raw = paths[stem].read_bytes()
        sha.update(stem.encode() + b"\0" + raw)
        bundle = json.loads(raw)
        for c in align_bundle(bundle):
            y = int(any(c.char_start < e and s < c.char_end for s, e in gold[stem]))
            items.append((stem, bundle, c, y))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    condition = {
        "dataset": {"version": args.dataset_version, "sha256": sha.hexdigest(), "bundle_ids": ids,
                    "n_bundles": len(ids), "n_claims": len(items),
                    "n_hallucinated": sum(it[3] for it in items),
                    "segmenter": "blocks/segmenter.py + blocks/aligner.py (align_bundle)"},
        "scorers": {n: scorer_stub_config(n, args) for n in scorer_names},
        "runs": args.runs,
        "code": git_info(),
    }
    man_path = out / "manifest.json"
    if man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        old = {k: manifest[k] for k in ("dataset", "runs", "code")}
        old_scorers = {n: manifest["scorers"].get(n, {}).get("condition") for n in scorer_names}
        new = {k: condition[k] for k in ("dataset", "runs", "code")}
        if old != new or old_scorers != condition["scorers"]:
            sys.exit("Refusing to resume: dataset / runs / code / scorer settings differ from "
                     f"{man_path}. Use a new --out folder for a new experimental condition.")
    else:
        manifest = {"run_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{args.dataset_version}_{args.provider}",
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "packages": packages(), "price_source": price_source,
                    **{k: condition[k] for k in ("dataset", "runs", "code")},
                    "scorers": {}, "rerun_policy": "no response cache; SelfCheckGPT samples reset before every rerun",
                    "budget_usd": args.max_cost_usd}
    if condition["code"]["git_dirty"]:
        print("WARNING: tracked files have uncommitted changes; results are not reportable (git_dirty=true).")

    rows_path = out / "scores.jsonl"
    done: dict[tuple, dict] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["method"], r["run"], r["bundle"], r["claim_id"])] = r  # last row wins
    spent = sum(r["cost_usd"] for r in done.values())
    n_resp = len(ids)
    print(f"{manifest['run_id']}: {n_resp} bundles, {len(items)} claims "
          f"({condition['dataset']['n_hallucinated']} hallucinated), {args.runs} run(s); "
          f"already spent ${spent:.4f} of ${args.max_cost_usd}", flush=True)

    summary = {"run_id": manifest["run_id"], "dataset": condition["dataset"], "threshold": args.threshold,
               "runs": args.runs, "scorers": {}}
    stopped = None
    with rows_path.open("a", encoding="utf-8") as fout:
        for name in scorer_names:
            scorer = make_scorer(name, args, price_in, price_out)
            method = scorer.name
            cfg = getattr(scorer, "config", {"method": method})
            manifest["scorers"][name] = {"method": method, "condition": condition["scorers"][name],
                                         "config": cfg}
            man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            est = EST_TOKENS.get((name, args.model))
            if est and price_in is not None:
                units = len(items) if name == "ragas" else n_resp
                estimate = args.runs * units * (est[0] * price_in + est[1] * price_out) / 1e6
            else:
                estimate = 0.0 if name == "presence" else None
            try:
                for run in range(1, args.runs + 1):
                    if hasattr(scorer, "reset_samples"):
                        scorer.reset_samples()  # fresh SelfCheckGPT samples every rerun
                    t0, n_new = time.perf_counter(), 0
                    for i, (stem, bundle, c, y) in enumerate(items):
                        key = (method, run, stem, f"c{c.char_start}")
                        if key in done and done[key]["status"] != "error":
                            continue
                        if spent >= args.max_cost_usd:
                            raise BudgetExceeded(f"budget ${args.max_cost_usd} reached (spent ${spent:.4f})")
                        if hasattr(scorer, "set_question"):
                            scorer.set_question(bundle["question"])
                        row = {"run_id": manifest["run_id"], "method": method, "run": run,
                               "bundle": stem, "claim_id": key[3], "char_start": c.char_start,
                               "char_end": c.char_end, "claim": c.claim_text,
                               "alignment": c.alignment, "y_true": y}
                        t_claim = time.perf_counter()
                        try:
                            res = scorer.score(c, bundle["citations"])
                            row.update(score=res.score, status=getattr(scorer, "last_status", "ok"),
                                       error=None, latency_ms=round(res.latency_ms, 1),
                                       cost_usd=res.cost_usd)
                        except Exception as e:  # noqa: BLE001 — recorded, never swallowed as 0
                            usage = getattr(scorer, "last_usage", {})
                            cost = 0.0
                            if usage and price_in is not None:
                                cost = (usage.get("prompt_tokens", 0) * price_in
                                        + usage.get("completion_tokens", 0) * price_out) / 1e6
                            row.update(score=None, status="error",
                                       error=f"{type(e).__name__}: {str(e)[:300]}",
                                       latency_ms=round((time.perf_counter() - t_claim) * 1000, 1),
                                       cost_usd=cost)
                            if type(e).__name__ in STOP_ERRORS:
                                stopped = row["error"]
                        u = getattr(scorer, "last_usage", None) or {}
                        row.update(prompt_tokens=u.get("prompt_tokens", 0),
                                   completion_tokens=u.get("completion_tokens", 0),
                                   llm_calls=u.get("calls", 0), http_requests=u.get("http_requests", 0),
                                   retries=max(0, u.get("http_requests", 0) - u.get("calls", 0)))
                        nli = getattr(scorer, "last_nli", None)
                        if nli is not None:
                            row.update(nli_pairs=nli["pairs"], nli_ms=round(nli["ms"], 1))
                        spent += row["cost_usd"]
                        done[key] = row
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fout.flush()
                        n_new += 1
                        if stopped:
                            raise BudgetExceeded(f"stopped on provider error: {stopped}")
                        if (i + 1) % 20 == 0:
                            print(f"  {method} run {run}: {i + 1}/{len(items)}  spent ${spent:.4f}", flush=True)
                    print(f"  {method} run {run} done ({n_new} new rows, "
                          f"{time.perf_counter() - t0:.0f}s, spent ${spent:.4f})", flush=True)
            except BudgetExceeded as e:
                stopped = str(e)
                print(f"STOP: {stopped}. Rerun the same command to resume.", flush=True)
            finally:
                if hasattr(scorer, "close"):
                    scorer.close()

            per_run, all_rows = [], []
            for run in range(1, args.runs + 1):
                rows = [done.get((method, run, it[0], f"c{it[2].char_start}")) for it in items]
                all_rows.extend(r for r in rows if r)
                ok = [r for r in rows if r and r["score"] is not None]
                m = {"run": run, "coverage": len(ok) / len(items) if items else 0.0}
                if ok:
                    ys = [r["y_true"] for r in ok]
                    m.update(prf(ys, [int(r["score"] < args.threshold) for r in ok]))
                    m["auroc"] = auroc(ys, [1 - r["score"] for r in ok])
                per_run.append(m)
            complete = all(m["coverage"] == 1.0 for m in per_run)
            f1s = [m["f1"] for m in per_run if "f1" in m]
            scores_by_claim = [[(done.get((method, r, it[0], f"c{it[2].char_start}")) or {}).get("score")
                                for r in range(1, args.runs + 1)] for it in items]
            changed = sum(len(set(col)) > 1 for col in scores_by_claim if None not in col)
            llm = {k: sum(r.get(k, 0) for r in all_rows)
                   for k in ("prompt_tokens", "completion_tokens", "llm_calls", "http_requests", "retries")}
            trusted = (args.runs >= 2 and complete and len(f1s) == args.runs
                       and (name != "selfcheck" or hasattr(scorer, "reset_samples")))
            summary["scorers"][method] = {
                "per_run": per_run,
                "complete": complete,
                "f1_mean": statistics.mean(f1s) if f1s else None,
                "det_var_f1_population": statistics.pvariance(f1s) if complete and len(f1s) > 1 else None,
                "det_var_f1_sample": statistics.variance(f1s) if complete and len(f1s) > 1 else None,
                "determinism_trusted": trusted,
                "claims_with_score_change": changed,
                "samples_reset": True if name == "selfcheck" else None,
                "llm_usage": llm if name != "presence" else None,
                "nli_usage": ({"model": cfg.get("nli_model"), "device": cfg.get("nli_device"),
                               "pairs": sum(r.get("nli_pairs", 0) for r in all_rows),
                               "ms": round(sum(r.get("nli_ms", 0) for r in all_rows), 1)}
                              if name == "selfcheck" else None),
                "cost_usd_estimate": estimate,
                "cost_usd_actual": sum(r["cost_usd"] for r in all_rows),
                "n_error_rows": sum(r["status"] == "error" for r in all_rows),
                "status_counts": {s: sum(r["status"] == s for r in all_rows)
                                  for s in sorted({r["status"] for r in all_rows})},
                "config": cfg,
            }
            if stopped:
                break

    summary["stopped"] = stopped
    summary["cost_usd_actual_total"] = spent
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("f1_mean", "det_var_f1_population", "determinism_trusted",
                                                "claims_with_score_change", "cost_usd_estimate",
                                                "cost_usd_actual")}
                      for k, v in summary["scorers"].items()}, indent=2))
    print(f"wrote {out}/manifest.json, scores.jsonl, summary.json" + (f"  [STOPPED: {stopped}]" if stopped else ""))


if __name__ == "__main__":
    main()
