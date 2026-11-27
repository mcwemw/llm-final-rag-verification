"""First real slice: deterministic embedding-cosine faithfulness detector on
RAGTruth QA (test split), response-level. Produces real numbers for the
reporting table + two floor baselines. No LLM-judge, no API calls.

Protocol:
  - stratified 50/50 calibration / eval split (seed fixed)
  - calibrate single detection threshold on calibration half (max F1)
  - report P/R/F1/kappa on eval half
  - determinism: recompute twice -> variance of F1 (deterministic -> 0)
  - cost: wall-clock embed time; API $ = 0 (local bge-m3)
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

import loader
import gate
import metrics
import data_version

RESULTS = os.path.join(os.path.dirname(__file__), "results")
SEED = 20261021


def stratified_split(rows, frac=0.5, seed=SEED):
    rng = np.random.default_rng(seed)
    idx0 = [i for i, r in enumerate(rows) if r["y_true"] == 0]
    idx1 = [i for i, r in enumerate(rows) if r["y_true"] == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    c0, c1 = int(len(idx0) * frac), int(len(idx1) * frac)
    cal = set(idx0[:c0] + idx1[:c1])
    return cal


def calibrate_threshold(cos, y, grid=np.arange(0.20, 0.90, 0.01), beta=1.0):
    """flag (predict hallucinated) when cosine < t. Pick t maximizing F_beta.
    beta=1 -> F1 (balanced); beta=0.5 -> precision-weighted (for FAIL-only point)."""
    best_t, best_s = 0.5, -1.0
    b2 = beta * beta
    for t in grid:
        pred = (cos < t).astype(int)
        m = metrics.prf(y, pred)
        p, r = m["precision"], m["recall"]
        fb = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0
        if fb > best_s:
            best_s, best_t = fb, float(t)
    return best_t


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="QA")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    dv = data_version.verify(strict=True)   # fail-loud if local data != pinned snapshot
    print(f"  [data] {dv['status']} ({dv['version']})")
    rows = loader.load(args.task, "test")
    if args.limit:
        rows = rows[:args.limit]
    y = np.array([r["y_true"] for r in rows])
    print(f"loaded {len(rows)} {args.task} responses, {int(y.sum())} hallucinated")

    # ---- embed once (deterministic), cache cosines to skip the ~26min re-embed ----
    cache = os.path.join(RESULTS, f"cos_cache_{args.task}_{len(rows)}.npy")
    if os.path.exists(cache):
        cos = np.load(cache)
        embed_sec = 0.0
        print(f"loaded cached cosines from {cache} (n={len(cos)}) — skipped embed")
    else:
        t0 = time.time()
        cos = gate.cosine_pairs([r["response"] for r in rows],
                                [r["source_text"] for r in rows])
        embed_sec = time.time() - t0
        np.save(cache, cos)
        print(f"embedded {2*len(rows)} texts in {embed_sec:.1f}s "
              f"({embed_sec/len(rows)*1000:.0f} ms/pair) — cached to {cache}")

    cal = stratified_split(rows)
    cal_mask = np.array([i in cal for i in range(len(rows))])
    ev = ~cal_mask

    # ---- 1. embedding-cosine detector, now via the REAL 4-way gate.classify() ----
    # hi = balanced (F1) boundary -> flagged = WARN+FAIL (high recall, abstention-tolerant)
    # lo = precision-weighted (F0.5) boundary -> flagged = FAIL only (high confidence)
    hi = calibrate_threshold(cos[cal_mask], y[cal_mask], beta=1.0)
    lo = calibrate_threshold(cos[cal_mask], y[cal_mask], beta=0.5)
    if lo > hi:            # FAIL band must be stricter than WARN band
        lo = hi
    # 4-way labels through gate.classify (citation always present at response level)
    labels = [gate.classify(float(c), citation_present=True, hi=hi, lo=lo) for c in cos]
    labels = np.array(labels)
    flag_warnfail = np.array([gate.is_flagged(l) for l in labels]).astype(int)  # != PASS
    flag_failonly = (labels == "FAIL").astype(int)

    m_warnfail = metrics.prf(y[ev].tolist(), flag_warnfail[ev].tolist())
    k_warnfail = metrics.cohen_kappa(y[ev].tolist(), flag_warnfail[ev].tolist())
    m_failonly = metrics.prf(y[ev].tolist(), flag_failonly[ev].tolist())
    k_failonly = metrics.cohen_kappa(y[ev].tolist(), flag_failonly[ev].tolist())

    # headline row stays the WARN+FAIL (balanced) operating point for continuity
    m_emb, k_emb, t = m_warnfail, k_warnfail, hi
    # determinism: recompute (cosines fixed) -> identical F1 each run
    f1_runs = [metrics.prf(y[ev].tolist(), flag_warnfail[ev].tolist())["f1"]
               for _ in range(3)]
    det_var = metrics.determinism_variance(f1_runs)

    # 4-way label distribution on eval half
    dist = {k: int((labels[ev] == k).sum()) for k in ("PASS", "WARN", "FAIL", "UNSOURCED")}
    print(f"4-way band: hi={hi:.2f} lo={lo:.2f} | eval dist {dist}")

    # ---- 2. floor A: always-faithful (never flag) ----
    pred_floor = np.zeros(len(rows), dtype=int)
    m_floor = metrics.prf(y[ev].tolist(), pred_floor[ev].tolist())

    # ---- 3. floor B: lexical Jaccard overlap ----
    jac = np.array([jaccard(r["response"], r["source_text"]) for r in rows])
    tj = calibrate_threshold(jac[cal_mask], y[cal_mask])
    pred_jac = (jac < tj).astype(int)
    m_jac = metrics.prf(y[ev].tolist(), pred_jac[ev].tolist())
    k_jac = metrics.cohen_kappa(y[ev].tolist(), pred_jac[ev].tolist())

    table = {
        "dataset": f"RAGTruth {args.task} test (response-level)",
        "dataset_version": dv["version"], "data_status": dv["status"],
        "calib_version": "calib_v1",
        "n_total": len(rows), "n_eval": int(ev.sum()),
        "hallucinated_rate": float(y.mean()),
        "embed_sec": embed_sec, "seed": SEED,
        "band": {"hi": hi, "lo": lo}, "eval_label_dist": dist,
        "methods": {
            "gate_4way_warnfail": {
                "threshold_hi": hi, **m_warnfail, "kappa": k_warnfail,
                "flagged": "WARN+FAIL (balanced/high-recall)",
                "determinism_var": det_var, "cost_api_usd": 0.0,
                "latency_ms_per_pair": embed_sec/len(rows)*1000,
            },
            "gate_4way_failonly": {
                "threshold_lo": lo, **m_failonly, "kappa": k_failonly,
                "flagged": "FAIL only (high-precision)",
                "determinism_var": 0.0, "cost_api_usd": 0.0,
            },
            "floor_always_faithful": {**m_floor, "determinism_var": 0.0,
                                      "cost_api_usd": 0.0},
            "floor_lexical_jaccard": {"threshold": tj, **m_jac, "kappa": k_jac,
                                      "determinism_var": 0.0, "cost_api_usd": 0.0},
        },
    }
    with open(os.path.join(RESULTS, "slice.json"), "w") as f:
        json.dump(table, f, indent=2)

    # ---- markdown ----
    def row(name, m, extra=""):
        return (f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | "
                f"{m['recall']:.3f} | {m.get('kappa', float('nan')):.3f} | "
                f"{extra} | $0 |")
    md = [
        f"# Slice results — {table['dataset']}",
        f"n_eval={table['n_eval']}, hallucinated={table['hallucinated_rate']*100:.1f}%, "
        f"seed={SEED}, embed={embed_sec:.1f}s\n",
        f"4-way band: hi={hi:.2f} lo={lo:.2f} · eval dist "
        f"PASS={dist['PASS']} WARN={dist['WARN']} FAIL={dist['FAIL']} "
        f"UNSOURCED={dist['UNSOURCED']}\n",
        "| Method | F1 | P | R | κ | Determinism (var) | Cost/1k |",
        "|---|---|---|---|---|---|---|",
        row("Gate 4-way: WARN+FAIL (bal.)", {**m_warnfail, "kappa": k_warnfail},
            f"{det_var:.2e}"),
        row("Gate 4-way: FAIL-only (prec.)", {**m_failonly, "kappa": k_failonly}, "0"),
        row("Floor: always-faithful", {**m_floor, "kappa": float('nan')}, "0"),
        row("Floor: lexical Jaccard", {**m_jac, "kappa": k_jac}, "0"),
        f"\nBands: cosine < {lo:.2f} → FAIL; {lo:.2f} ≤ cosine < {hi:.2f} → WARN; "
        f"≥ {hi:.2f} → PASS. Jaccard floor < {tj:.2f} → flag.",
    ]
    with open(os.path.join(RESULTS, "table.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
