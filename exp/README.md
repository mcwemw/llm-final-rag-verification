# exp/ — Benchmark Harness (zone C)

Common evaluation harness that scores each published verification technique against
RAGTruth human labels and reports F1 / κ / determinism / cost (proposal §4, Figure 1 zone C).

## Files
- `run_harness.py` — main entry; runs techniques over a dataset, emits comparison table.
- `run_slice.py` — single-slice runner.
- `gate.py` — FAIL/WARN/PASS decision gate.
- `loader.py` — dataset loader (RAGTruth / calib subset).
- `metrics.py` — P/R/F1 / Cohen's κ / determinism variance.
- `scorers/` — `nli_scorer.py` (cross-encoder NLI, deterministic), `llm_judge.py` (RAGAS-style, Groq), `selfcheck.py` (SelfCheckGPT-NLI, sampling consistency).
- `thresholds.json` — released calibration thresholds (proposal §3.3, reproducibility).

## Dependencies (⚠️ read before running)
- **Data**: requires `data/frozen/calib_v1/` + RAGTruth split. This PR assumes those are
  restored on `main` first (see `fix/restore-calib-v1`). Harness binds to a calibration
  **dataset version** (`manifest.json` sha) — do not run against an unversioned subset.
- **Keys**: LLM-judge / SelfCheck need `GROQ_API_KEY` (`source ~/.zsh_secrets`). Never commit keys.
- **Model note**: `llama-3.3-70b-versatile` retired 2026-08-16. Any run must record the live
  model in output; results from the retired model are void (do not treat new-model results as
  a continuation of old).

## Reproduce
```bash
source ~/.zsh_secrets
python run_harness.py --techniques all --llm-limit 100 --sc-limit 50 --sc-samples 3
```

## Hygiene (enforced by .gitignore)
`exp/results/`, `exp/data/`, `__pycache__/` are never committed. Scores are exchanged via the
`score-export-v1` contract (repo root `score-export-v1.schema.md`), validated by the harness
(fail-loud on missing `score_direction` / version fields).
