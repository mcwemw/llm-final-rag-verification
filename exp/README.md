# exp/ — Benchmark Harness (zone C)

Common evaluation harness that scores each published faithfulness-verification technique
against RAGTruth human labels and reports F1 / κ / determinism / cost (proposal §4, Figure 1
zone C). The harness is the integration contribution; every scorer is a published technique.

## Files
- `run_harness.py` — main entry; runs techniques over a dataset, emits the comparison table.
- `run_slice.py` — single-slice runner.
- `gate.py` — FAIL/WARN/PASS decision gate (bge-m3 cosine).
- `loader.py` — RAGTruth loader (response-level y_true).
- `metrics.py` — P/R/F1 / Cohen's κ / determinism variance.
- `score_export.py` — `score-export-v1` validator + normalization (consumes external scorer JSON).
- `scorers/` — `nli_scorer.py` (cross-encoder NLI, deterministic), `llm_judge.py`
  (RAGAS-style, Groq), `selfcheck.py` (SelfCheckGPT-NLI + stateful `SelfCheckScorer`),
  `base.py` (`QuestionAwareScorer`: `set_question` / `reset_samples`).
- `thresholds.json` — released calibration thresholds (proposal §3.3).
- `test_*.py` — unit + integration + e2e tests.

## External scorer integration (score-export-v1)
A scorer run elsewhere (own key) exports per-response scores as JSON per
`../score-export-v1.schema.md`; the harness consumes it:

```bash
python run_harness.py --score-export path/to/export.json --dataset-version calib_v1
```

Validation is fail-loud: missing `score_direction` / `schema_version` mismatch /
(for selfcheck) missing `samples_reset` / a null score without an `error` → rejected.
Scores are normalized to P(hallucinated); claim-level scores are retained for audit;
response-level aggregation = max over claims (conservative).

## Language support
`--lang zh` support for Chinese segmentation/alignment lives in `../blocks/` (separate change).

## Dependencies & hygiene
- Data: requires `data/frozen/calib_v1/` + RAGTruth split (NOT committed here; gitignored).
  The harness binds to a calibration dataset version — do not run against an unversioned subset.
- Keys: LLM-judge / SelfCheck need `GROQ_API_KEY` in the environment. Never commit keys.
- Model: `llama-3.3-70b-versatile` retired 2026-08-16 → default is a live model
  (`GROQ_JUDGE_MODEL`, fallback `openai/gpt-oss-20b`); every run records the model used.
- `results/`, `data/`, `__pycache__/` are never committed (see `.gitignore`).

## Reproduce
```bash
source ~/.zsh_secrets    # provides GROQ_API_KEY
python run_harness.py --techniques all --llm-limit 100 --sc-limit 50 --sc-samples 3
```
