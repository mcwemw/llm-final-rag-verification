"""Convert a tools/run_baselines.py run directory into score-export-v1 files for the harness.

Usage:
    python -m tools.export_scores --run-dir ../runs/pilot [--out ../runs/pilot/export]

Writes one JSON per (technique x rerun): {technique}_run{k}.json, following
score-export-v1.schema.md (feat/harness). Required fields are asserted the same way the harness
does (fail-loud). Extra fields — llm_usage, nli, cost, coverage, char offsets, status — are
additions proposed to the harness owner; a strict reader can ignore them.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "score-export-v1"
TECHNIQUE = {"presence-only": "presence", "ragas-llm-judge": "ragas", "selfcheckgpt-nli": "selfcheck"}
REQUIRED = ["schema_version", "run_id", "technique", "model", "scorer_version", "dataset_version",
            "score_direction", "score_range", "rerun", "created_at", "scores"]
REQUIRED_SELFCHECK = ["n_samples", "generator_model", "samples_reset"]
BUNDLE_PREFIX = "ragtruth-"


def response_id(bundle: str) -> str:
    """score-export-v1 bundle_id = bare RAGTruth response id ("ragtruth-12003" -> "12003").

    The harness joins on str(row["id"]); dataset identity is carried by dataset_version and
    dataset_sha, so the prefix adds nothing. Agreed with the harness owner 2026-11-27.
    """
    return bundle[len(BUNDLE_PREFIX):] if bundle.startswith(BUNDLE_PREFIX) else bundle


def validate(doc: dict) -> None:
    """Same checks the harness performs on read (score-export-v1 §2-3)."""
    missing = [k for k in REQUIRED if k not in doc]
    if doc.get("technique") == "selfcheck":
        missing += [k for k in REQUIRED_SELFCHECK if k not in doc]
    assert not missing, f"missing required fields: {missing}"
    assert doc["schema_version"] == SCHEMA
    assert doc["score_direction"] in ("high_means_faithful", "high_means_hallucinated")
    lo, hi = doc["score_range"]
    for s in doc["scores"]:
        assert {"bundle_id", "claim_id", "score", "error"} <= s.keys(), s
        if s["score"] is None:
            assert s["error"], f"null score without error: {s}"  # error is never swallowed
        else:
            assert lo <= s["score"] <= hi, s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <run-dir>/export")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows: dict[tuple, dict] = {}
    for line in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[(r["method"], r["run"], r["bundle"], r["claim_id"])] = r  # last row wins
    out = Path(args.out) if args.out else run_dir / "export"
    out.mkdir(parents=True, exist_ok=True)

    ds, code = manifest["dataset"], manifest["code"]
    written = []
    for name, info in manifest["scorers"].items():
        method, cfg = info["method"], info["config"]
        tech = TECHNIQUE[method]
        s_sum = summary["scorers"].get(method, {})
        for run in range(1, manifest["runs"] + 1):
            rs = [r for (m, k, _, _), r in rows.items() if m == method and k == run]
            rs.sort(key=lambda r: (r["bundle"], r["char_start"]))
            if not rs:
                continue
            llm = {k: sum(r.get(k, 0) for r in rs)
                   for k in ("prompt_tokens", "completion_tokens", "llm_calls", "http_requests", "retries")}
            doc = {
                "schema_version": SCHEMA,
                "run_id": f"{manifest['run_id']}_{tech}_run{run}",
                "technique": tech,
                # judge / decision model: RAGAS -> LLM judge; SelfCheckGPT -> NLI model
                "model": {"ragas": cfg.get("model"), "selfcheck": cfg.get("nli_model"),
                          "presence": "none (citation presence)"}[tech],
                "scorer_version": f"blocks/baselines.py@{code['git_sha'][:12]}"
                                  + ("-dirty" if code["git_dirty"] else ""),
                "dataset_version": ds["version"],
                "dataset_sha": ds["sha256"],
                "score_direction": "high_means_faithful",
                "score_range": [0.0, 1.0],
                "rerun": run > 1,
                "rerun_index": run,
                "n_reruns": manifest["runs"],
                "determinism_trusted": bool(s_sum.get("determinism_trusted")),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "provider": cfg.get("provider"),
                "base_url": cfg.get("base_url"),
                "temperature": cfg.get("temperature"),
                "top_p": cfg.get("top_p"),
                "seed": cfg.get("seed"),
                "prompt_version": cfg.get("prompt_version"),
                "price_in_per_mtok": cfg.get("price_in_per_mtok"),
                "price_out_per_mtok": cfg.get("price_out_per_mtok"),
                "price_source": manifest.get("price_source"),
                "packages": manifest.get("packages"),
                "coverage": sum(r["score"] is not None for r in rs) / ds["n_claims"],
                "llm_usage": llm if tech != "presence" else None,
                "cost_usd_actual": sum(r["cost_usd"] for r in rs),
                "cost_usd_estimate_all_runs": s_sum.get("cost_usd_estimate"),
                "scores": [{"bundle_id": response_id(r["bundle"]), "bundle": r["bundle"], "claim_id": r["claim_id"], "score": r["score"],
                            "error": r["error"], "status": r["status"], "char_start": r["char_start"],
                            "char_end": r["char_end"], "cost_usd": r["cost_usd"],
                            "retries": r.get("retries", 0)} for r in rs],
            }
            if tech == "selfcheck":
                doc.update(n_samples=cfg.get("n_samples"), generator_model=cfg.get("generator_model"),
                           samples_reset=True, sampling_unit=cfg.get("sampling_unit"),
                           nli={"model": cfg.get("nli_model"), "device": cfg.get("nli_device"),
                                "transformers": (manifest.get("packages") or {}).get("transformers"),
                                "pairs": sum(r.get("nli_pairs", 0) for r in rs),
                                "ms": round(sum(r.get("nli_ms", 0) for r in rs), 1)})
            validate(doc)
            path = out / f"{tech}_run{run}.json"
            path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            written.append(path.name)
    print(f"wrote {len(written)} score-export-v1 files to {out}: {', '.join(written)}")


if __name__ == "__main__":
    main()
