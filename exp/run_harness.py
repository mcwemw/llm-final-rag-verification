"""Full faithfulness harness — produces the multi-technique comparison table.

Techniques evaluated (all published, harness is the integration contribution):
  1. Gate 4-way WARN+FAIL  — bge-m3 cosine (this module)   determinism_var = 0
  2. Gate 4-way FAIL-only  — bge-m3 cosine, high-precision  determinism_var = 0
  3. NLI entailment        — cross-encoder/nli-deberta-v3   determinism_var = 0
  4. LLM-judge RAGAS-style — Groq llama-3.3-70b             determinism_var > 0  ← key measurement
  5. Floor: always-faithful
  6. Floor: lexical Jaccard

For techniques 1-4 the eval protocol is identical:
  - stratified 50/50 calibration / eval split (SEED=20261021)
  - calibrate detection threshold on calibration half (max-F1)
  - report P/R/F1/κ on eval half
  - determinism: n_runs identical runs, variance of F1

Usage:
  # embedding-cosine only (no network, no extra deps):
  python run_harness.py --techniques emb

  # + NLI (needs transformers, slow on CPU):
  python run_harness.py --techniques emb,nli

  # + LLM-judge (needs GROQ_API_KEY, source ~/.zsh_secrets first):
  source ~/.zsh_secrets && python run_harness.py --techniques emb,nli,llm --llm-limit 100

  # all, limit LLM to 100 records for speed:
  source ~/.zsh_secrets && python run_harness.py --techniques all --llm-limit 100
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np

# add exp/ to path so relative imports work
sys.path.insert(0, os.path.dirname(__file__))
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
    rng.shuffle(idx0); rng.shuffle(idx1)
    c0, c1 = int(len(idx0) * frac), int(len(idx1) * frac)
    return set(idx0[:c0] + idx1[:c1])


def calibrate_t(scores, y, beta=1.0):
    b2 = beta * beta
    best_t, best_s = 0.5, -1.0
    for t in np.arange(0.05, 0.98, 0.01):
        pred = (scores < t).astype(int)
        m = metrics.prf(y.tolist(), pred.tolist())
        p, r = m["precision"], m["recall"]
        fb = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0
        if fb > best_s:
            best_s, best_t = fb, float(t)
    return best_t


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


def _row(name, m, det_var, cost_usd, latency_ms="—"):
    return {
        "name": name,
        "f1": round(m["f1"], 4), "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4), "kappa": round(m.get("kappa", float("nan")), 4),
        "determinism_var": det_var, "cost_usd_per_1k": cost_usd,
        "latency_ms_per_pair": latency_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="QA")
    ap.add_argument("--limit", type=int, default=0, help="0 = all test rows")
    ap.add_argument("--techniques", default="emb",
                    help="comma-sep: emb,nli,llm  or 'all'")
    ap.add_argument("--llm-limit", type=int, default=100,
                    help="number of records to run through LLM-judge (cost control)")
    ap.add_argument("--llm-runs", type=int, default=3,
                    help="LLM-judge repeat runs for determinism measurement")
    ap.add_argument("--llm-model", default="llama-3.3-70b-versatile")
    ap.add_argument("--sc-limit", type=int, default=50,
                    help="SelfCheckGPT record cap (regenerates N samples each)")
    ap.add_argument("--sc-samples", type=int, default=3,
                    help="SelfCheckGPT stochastic samples per record")
    ap.add_argument("--out", default=os.path.join(RESULTS, "harness.json"))
    args = ap.parse_args()

    techs = set(args.techniques.lower().split(","))
    if "all" in techs:
        techs = {"emb", "nli", "llm", "selfcheck"}

    os.makedirs(RESULTS, exist_ok=True)
    dv = data_version.verify(strict=True)   # fail-loud if local data != pinned snapshot
    print(f"  [data] {dv['status']} ({dv['version']})")
    rows = loader.load(args.task, "test")
    if args.limit:
        rows = rows[:args.limit]
    y = np.array([r["y_true"] for r in rows])
    print(f"Loaded {len(rows)} {args.task} test responses, "
          f"{int(y.sum())} hallucinated ({y.mean()*100:.1f}%)")

    cal_set = stratified_split(rows)
    cal_mask = np.array([i in cal_set for i in range(len(rows))])
    ev = ~cal_mask
    n_eval = int(ev.sum())
    print(f"Split: {int(cal_mask.sum())} calibration, {n_eval} eval")

    result_rows = []

    # ---- 1 & 2: embedding-cosine gate ----------------------------------------
    if "emb" in techs:
        cache = os.path.join(RESULTS, f"cos_cache_{args.task}_{len(rows)}.npy")
        if os.path.exists(cache):
            cos = np.load(cache)
            embed_sec = 0.0
            print(f"  [emb] loaded cosine cache ({len(cos)} pairs)")
        else:
            t0 = time.time()
            cos = gate.cosine_pairs([r["response"] for r in rows],
                                    [r["source_text"] for r in rows])
            embed_sec = time.time() - t0
            np.save(cache, cos)
            print(f"  [emb] embedded in {embed_sec:.1f}s — cached")

        hi = calibrate_t(cos[cal_mask], y[cal_mask], beta=1.0)
        lo = calibrate_t(cos[cal_mask], y[cal_mask], beta=0.5)
        if lo > hi: lo = hi

        labels = np.array([gate.classify(float(c), True, hi=hi, lo=lo) for c in cos])
        fw = (labels != "PASS").astype(int)   # WARN+FAIL
        ff = (labels == "FAIL").astype(int)   # FAIL only

        mw = metrics.prf(y[ev].tolist(), fw[ev].tolist())
        mw["kappa"] = metrics.cohen_kappa(y[ev].tolist(), fw[ev].tolist())
        mf = metrics.prf(y[ev].tolist(), ff[ev].tolist())
        mf["kappa"] = metrics.cohen_kappa(y[ev].tolist(), ff[ev].tolist())

        # determinism: recompute from cached cosines, identical each time
        f1_runs = [metrics.prf(y[ev].tolist(), fw[ev].tolist())["f1"] for _ in range(3)]
        det_var = metrics.determinism_variance(f1_runs)

        lat = (embed_sec / len(rows) * 1000) if embed_sec else "cached"
        result_rows.append(_row("Gate bge-m3 WARN+FAIL (bal.)", mw, det_var, 0.0, lat))
        result_rows.append(_row("Gate bge-m3 FAIL-only (prec.)", mf, 0.0, 0.0))

        dist = {k: int((labels[ev] == k).sum()) for k in ("PASS","WARN","FAIL","UNSOURCED")}
        print(f"  [emb] hi={hi:.3f} lo={lo:.3f} | eval dist {dist}")
        print(f"  [emb] WARN+FAIL  F1={mw['f1']:.3f} κ={mw['kappa']:.3f} det_var={det_var:.2e}")

    # ---- 3: NLI entailment ---------------------------------------------------
    if "nli" in techs:
        print("\n[nli] loading cross-encoder/nli-deberta-v3-small…")
        nli_cache = os.path.join(RESULTS, f"nli_cache_{args.task}_{len(rows)}.npy")
        if os.path.exists(nli_cache):
            nli_scores = np.load(nli_cache)
            print(f"  [nli] loaded cache ({len(nli_scores)} scores)")
        else:
            from scorers.nli_scorer import score_batch as nli_batch
            t0 = time.time()
            nli_scores = np.array(nli_batch(rows))
            nli_sec = time.time() - t0
            np.save(nli_cache, nli_scores)
            print(f"  [nli] scored in {nli_sec:.1f}s — cached")

        # NLI: low entailment = hallucinated (flag when score < t)
        nt = calibrate_t(nli_scores[cal_mask], y[cal_mask])
        pred_nli = (nli_scores < nt).astype(int)
        mnli = metrics.prf(y[ev].tolist(), pred_nli[ev].tolist())
        mnli["kappa"] = metrics.cohen_kappa(y[ev].tolist(), pred_nli[ev].tolist())

        # determinism: NLI is deterministic, recompute from cache
        f1_nli_runs = [metrics.prf(y[ev].tolist(), pred_nli[ev].tolist())["f1"] for _ in range(3)]
        det_nli = metrics.determinism_variance(f1_nli_runs)

        result_rows.append(_row("NLI cross-encoder (DeBERTa-v3)", mnli, det_nli, 0.0))
        print(f"  [nli] t={nt:.3f} F1={mnli['f1']:.3f} κ={mnli['kappa']:.3f} det_var={det_nli:.2e}")

    # ---- 4: LLM-judge (RAGAS-style) -----------------------------------------
    if "llm" in techs:
        from scorers.llm_judge import score_batch as llm_batch, threshold_scores as llm_thresh

        llm_rows = rows[:args.llm_limit]
        y_llm = y[:args.llm_limit]
        cal_llm = stratified_split(llm_rows)
        cal_llm_mask = np.array([i in cal_llm for i in range(len(llm_rows))])
        ev_llm = ~cal_llm_mask

        print(f"\n[llm] scoring {len(llm_rows)} records × {args.llm_runs} runs with {args.llm_model}…")
        llm_cache = os.path.join(RESULTS, f"llm_scores_{args.task}_{len(llm_rows)}_r{args.llm_runs}.json")

        if os.path.exists(llm_cache):
            with open(llm_cache) as f:
                all_runs = json.load(f)
            print(f"  [llm] loaded cache ({len(all_runs)} runs × {len(all_runs[0])} records)")
        else:
            t0 = time.time()
            all_runs = llm_batch(llm_rows, model=args.llm_model,
                                 n_runs=args.llm_runs, temperature=0.3)
            llm_sec = time.time() - t0
            with open(llm_cache, "w") as f:
                json.dump(all_runs, f)
            print(f"  [llm] finished {args.llm_runs} runs in {llm_sec:.1f}s — cached")

        # calibrate threshold on first run only (calibration half)
        s0 = np.array([v if v is not None else 0.5 for v in all_runs[0]])
        # LLM-judge: low faithfulness_score = hallucinated
        lt = calibrate_t(s0[cal_llm_mask], y_llm[cal_llm_mask])

        # score each run → binary predictions → F1
        f1_llm_runs = []
        preds_per_run = []
        for run_scores in all_runs:
            s = np.array([v if v is not None else 0.5 for v in run_scores])
            pred = (s < lt).astype(int)
            preds_per_run.append(pred)
            f1_llm_runs.append(metrics.prf(y_llm[ev_llm].tolist(),
                                           pred[ev_llm].tolist())["f1"])

        det_llm = metrics.determinism_variance(f1_llm_runs)

        # aggregate metrics: majority vote across runs
        votes = np.stack(preds_per_run, axis=0)   # (n_runs, n_records)
        pred_maj = (votes.mean(axis=0) >= 0.5).astype(int)
        mllm = metrics.prf(y_llm[ev_llm].tolist(), pred_maj[ev_llm].tolist())
        mllm["kappa"] = metrics.cohen_kappa(y_llm[ev_llm].tolist(),
                                             pred_maj[ev_llm].tolist())

        cost_per_1k = 0.59 / 1000  # llama-3.3-70b on Groq: ~$0.59/1M tokens → ~$0.59/1k pairs
        result_rows.append({
            **_row(f"LLM-judge RAGAS-style (Groq {args.llm_model})", mllm, det_llm,
                   cost_per_1k, "~1500ms"),
            "f1_per_run": [round(f, 4) for f in f1_llm_runs],
            "note": f"n={len(llm_rows)} records (cost-capped); majority vote over {args.llm_runs} runs",
        })
        print(f"  [llm] t={lt:.3f} maj-F1={mllm['f1']:.3f} κ={mllm['kappa']:.3f}")
        print(f"  [llm] F1 per run: {[round(f,3) for f in f1_llm_runs]} → det_var={det_llm:.2e}")

    # ---- 5: SelfCheckGPT-NLI (sampling consistency) -------------------------
    if "selfcheck" in techs:
        from scorers.selfcheck import score_batch as sc_batch

        sc_rows = rows[:args.sc_limit]
        y_sc = y[:args.sc_limit]
        cal_sc = stratified_split(sc_rows)
        cal_sc_mask = np.array([i in cal_sc for i in range(len(sc_rows))])
        ev_sc = ~cal_sc_mask

        sc_cache = os.path.join(RESULTS,
            f"selfcheck_{args.task}_{len(sc_rows)}_s{args.sc_samples}_r{args.llm_runs}.json")
        if os.path.exists(sc_cache):
            with open(sc_cache) as f:
                sc_runs = json.load(f)
            print(f"  [selfcheck] loaded cache ({len(sc_runs)} runs × {len(sc_runs[0])})")
        else:
            print(f"\n[selfcheck] {len(sc_rows)} records × {args.sc_samples} samples "
                  f"× {args.llm_runs} runs ({args.llm_model})…")
            t0 = time.time()
            sc_runs = sc_batch(sc_rows, model=args.llm_model, n_samples=args.sc_samples,
                               n_runs=args.llm_runs, temperature=1.0)
            with open(sc_cache, "w") as f:
                json.dump(sc_runs, f)
            print(f"  [selfcheck] finished in {time.time() - t0:.1f}s — cached")

        s0 = np.array(sc_runs[0])
        st = calibrate_t(s0[cal_sc_mask], y_sc[cal_sc_mask])  # low consistency = hallucinated
        f1_sc_runs, preds_sc = [], []
        for run_scores in sc_runs:
            s = np.array(run_scores)
            pred = (s < st).astype(int)
            preds_sc.append(pred)
            f1_sc_runs.append(metrics.prf(y_sc[ev_sc].tolist(), pred[ev_sc].tolist())["f1"])
        det_sc = metrics.determinism_variance(f1_sc_runs)
        votes = np.stack(preds_sc, axis=0)
        pred_maj = (votes.mean(axis=0) >= 0.5).astype(int)
        msc = metrics.prf(y_sc[ev_sc].tolist(), pred_maj[ev_sc].tolist())
        msc["kappa"] = metrics.cohen_kappa(y_sc[ev_sc].tolist(), pred_maj[ev_sc].tolist())
        cost_per_1k = 0.59 / 1000 * args.sc_samples  # N samples/record on Groq
        result_rows.append({
            **_row(f"SelfCheckGPT-NLI (Groq, {args.sc_samples} samples)", msc, det_sc,
                   cost_per_1k, "~4000ms"),
            "f1_per_run": [round(f, 4) for f in f1_sc_runs],
            "note": f"n={len(sc_rows)} records (cost-capped); {args.sc_samples} samples/rec; "
                    f"majority vote over {args.llm_runs} runs",
        })
        print(f"  [selfcheck] t={st:.3f} maj-F1={msc['f1']:.3f} κ={msc['kappa']:.3f} "
              f"det_var={det_sc:.2e}")

    # ---- floors --------------------------------------------------------------
    if "emb" in techs:   # floors use same split as emb
        pred_af = np.zeros(len(rows), dtype=int)
        m_af = metrics.prf(y[ev].tolist(), pred_af[ev].tolist())
        result_rows.append(_row("Floor: always-faithful", m_af, 0.0, 0.0))

        jac_s = np.array([jaccard(r["response"], r["source_text"]) for r in rows])
        tj = calibrate_t(jac_s[cal_mask], y[cal_mask])
        pred_jac = (jac_s < tj).astype(int)
        m_jac = metrics.prf(y[ev].tolist(), pred_jac[ev].tolist())
        m_jac["kappa"] = metrics.cohen_kappa(y[ev].tolist(), pred_jac[ev].tolist())
        result_rows.append(_row("Floor: lexical Jaccard", m_jac, 0.0, 0.0))

    # ---- save & print table --------------------------------------------------
    summary = {
        "dataset": f"RAGTruth {args.task} test (response-level)",
        "dataset_version": dv["version"],
        "data_status": dv["status"],
        "calib_version": "calib_v1",
        "n_total": len(rows), "n_eval": n_eval,
        "hallucinated_rate": float(y.mean()),
        "seed": SEED,
        "methods": result_rows,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ saved {args.out}")

    # markdown table
    md_lines = [
        f"# Harness results — {summary['dataset']}",
        f"n_eval={n_eval}, hallucinated={y.mean()*100:.1f}%, seed={SEED}\n",
        "| Method | F1 | P | R | κ | Det-var | Cost/1k |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result_rows:
        dv = r["determinism_var"]
        dv_s = "0" if dv == 0.0 else f"{dv:.2e}"
        md_lines.append(
            f"| {r['name']} | {r['f1']:.3f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r.get('kappa', float('nan')):.3f} | "
            f"{dv_s} | ${r['cost_usd_per_1k']:.4f} |"
        )
    md = "\n".join(md_lines)
    out_md = os.path.join(RESULTS, "harness_table.md")
    with open(out_md, "w") as f:
        f.write(md)
    print(f"\n{md}")


if __name__ == "__main__":
    main()
