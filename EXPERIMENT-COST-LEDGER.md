# Experiment Cost Ledger / 實驗成本帳本

This ledger is the formal, traceable record of the project's research cost. It is not an estimate sheet.

Every paid or quota-consuming API run gets a row. Amounts are in **USD**. They are computed from the token counts the API returned and the provider's list price, and they are checked against the provider's billing page (column *Evidence*).

**Three states, never mixed:**

| State | Meaning |
|---|---|
| **A. Spent / 已實際支出** | The API calls have happened, and the cost is real. |
| **B. Approved, not spent / 已核准但未支出** | A budget line approved by the team lead. Nothing has been spent yet, and it is **not** counted as spent. |
| **C. Pending approval / 尚待核准的預估支出** | An estimate only. Do not run it until it is approved and moved to B. |

**Rules:**
- A paid run starts only against an approved B line.
- The runner's `--max-cost-usd` is set to the remaining amount of that line, so it stops before going over.
- After the run, the actual cost goes to A, and the B line is reduced by that amount.
- Payer: the executor fronts the cost, and it is shared later under **§6a** (judge-selection decision record, budget principle).
- Prices are list prices, uncached, read on the date shown (sources at the bottom).
- Run directories (`runs/<dir>/`: `manifest.json`, `scores.jsonl`, `judge_calls.jsonl`, `summary.json`) are kept outside the repo. They hold corpus text and raw judge output. They are available on request.

---

## A. Spent / 已實際支出

| # | Date (UTC) | Stage | Provider / model | Dataset / manifest SHA-256 | Units × reruns | Context | Input tok | Output tok | Reasoning tok | Cost (USD) | Cumulative (USD) | Executor / payer | Evidence | Status / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 2026-11-26 | Smoke | Fireworks `glm-5p3-flash` vs `deepseek-v4p1-flash` | calib_v1 (1 response) | 6 claims × 1 | cited_span | not logged | not logged | not logged | ≈0.024 | 0.024 | Jimmy / Jimmy | Fireworks billing page (read 2026-11-27): DeepSeek V4.1 Flash **$0.02** | Spent, to be shared (§6a). Model comparison before per-call logging existed. The DeepSeek part ($0.02) is the **billed** amount, the only DeepSeek use. The GLM part (≈$0.004) is estimated from the pilot-2 unit cost and is included in the billed GLM total. |
| A2 | 2026-11-26 18:26 | Pilot (format) | Fireworks `accounts/fireworks/models/glm-5p3-flash` | calib_v1 · run-dataset `04f2f570…cab3b3` | 3 responses / 15 claims × 3 (presence, ragas, selfcheck) | cited_span | 93,926 | 54,766 | not recorded¹ | 0.0415 | 0.0655 | Jimmy / Jimmy | `runs/pilot_fireworks_glm/scores.jsonl`, run `20261126T182622Z_calib_v1_fireworks` | Spent, to be shared (§6a). Pilot 1. 2/45 RAGAS claims were truncated, which led to max_tokens=4096. GLM has since been excluded, so this is a format check, not a result. |
| A3 | 2026-11-26 18:40 | Pilot (format) | Fireworks `accounts/fireworks/models/glm-5p3-flash` | calib_v1 · run-dataset `04f2f570…cab3b3` | 3 responses / 15 claims × 3 (presence, ragas, selfcheck) | cited_span | 86,307 | 47,270 | not recorded¹ | 0.0366 | 0.1021 | Jimmy / Jimmy | `runs/pilot2_fireworks_glm/scores.jsonl`, run `20261126T184015Z_calib_v1_fireworks` | Spent, to be shared (§6a). Pilot 2. Its 9 exports are now format fixtures in `exp/fixtures/` (PR #11). |
| A4 | 2026-11-27 | Smoke | Fireworks `muse-spark-open`, `inkling-2-small` | — | 4 calls (JSON / plain × 2 models) | — | 382 | 813 | 761 | 0.0034 | 0.1055 | Jimmy / Jimmy | call responses (usage) in session log; Fireworks billing page | Spent, to be shared (§6a). Gate 1 connection and JSON-mode smoke test. Muse with max_tokens=400 spent all tokens on reasoning and returned empty content. |
| A5 | 2026-11-27 10:16 | Pilot (single-answer cost test) | Fireworks `accounts/fireworks/models/muse-spark-open` | calib_v1.1 manifest `fe67dfe1…d6b0ee` · run-dataset `9681014d…71cb0f7` | 1 response / 9 claims × 1 (ragas) | cited_span | 18,065 | 15,869 | 13,714 | 0.0900 | 0.1955 | Jimmy / Jimmy | `runs/t_muse/scores.jsonl`, `judge_calls.jsonl`, run `20261127T101640Z_calib_v1.1_fireworks` | Spent, to be shared (§6a). Measured per-claim cost: $0.0100. |
| A6 | 2026-11-27 10:19 | Pilot (single-answer cost test) | Fireworks `accounts/fireworks/models/muse-spark-open` | calib_v1.1 manifest `fe67dfe1…d6b0ee` · run-dataset `9681014d…71cb0f7` | 1 response / 9 claims × 1 (ragas) | all_docs | 20,629 | 19,722 | 17,338 | 0.1096 | 0.3051 | Jimmy / Jimmy | `runs/t_muse_all/scores.jsonl`, `judge_calls.jsonl`, run `20261127T101936Z_calib_v1.1_fireworks` | Spent, to be shared (§6a). Context comparison on the same answer. Rows include 2 claims re-run after a local timeout (resume). |
| A7 | 2026-11-27 | Smoke (cost optimisation) | Fireworks `muse-spark-open`, `inkling-2-small` | — | 6 calls (`reasoning_effort` default / low / medium × 2 models) | — | 1,125 | 7,273 | 6,411 | 0.0228 | 0.3279 | Jimmy / Jimmy | call responses (usage) in session log; Fireworks billing page | Spent, to be shared (§6a). `reasoning_effort=low` cuts reasoning tokens: Muse −69%, Inkling −65%. Using it would be a protocol change and is not approved. |
| — | 2026-11-26 – 27 | Pilot / Smoke | Groq `openai/gpt-oss-20b` (free tier) | calib_v1 | 199 claims × up to 3 | cited_span | — | — | — | **0.00** | 0.3279 | Jimmy / — | Groq free tier, no charge | Preliminary runs only, not results. The runner logged a nominal $0.0213 at third-party prices, but the free tier does not charge. |
| — | 2026-11-27 – | Smoke / Pilot | NVIDIA NIM `mistral-large-4-1t-a52b`, `nemotron-3-ultra-550b-a55b` | calib_v1.1 manifest `fe67dfe1…d6b0ee` | Nemotron: 207 claims × 3 × 2 contexts (running) | cited_span, all_docs | logged per run | logged per run | logged per run | **0.00** | 0.3279 | Jimmy / — | NIM hosted API, no charge observed | Free-tier terms, RPM and data retention are **UNKNOWN**. Free is not assumed to be permanent. |

**Total spent (A), billed: US$0.29** (Fireworks billing page, November, rounded to cents; see the reconciliation below). The token-based computation in the rows is $0.328. It is higher because it prices every input token at the uncached list price.

### Billing reconciliation (Fireworks Billing & Usage, Spend, read 2026-11-27)

| Model | Billed: input (uncached) / output / total | Ledger (tokens × list price): input / output / total | Rows |
|---|---|---|---|
| Muse Spark Open | $0.02 / $0.18 / **$0.20** | $0.050 / $0.173 / $0.223 | A4, A5, A6, A7 |
| GLM 5.3 Flash | $0.01 / $0.05 / **$0.07** | $0.027 / $0.051 / ≈$0.082 | A1 (GLM part), A2, A3 |
| DeepSeek V4.1 Flash | **$0.02** | (not logged) | A1 |
| Inkling 2 Small | not listed | $0.003 | A4, A7 |
| **Total** | **$0.29** | $0.328 | |

**Findings.**
- **Output agrees** with the token-based figure: Muse $0.18 vs $0.173, GLM $0.05 vs $0.051.
- **Input is billed at about 40% of list.** The page shows only "Input (uncached)". A likely cause is a cached-prefix discount on the repeated RAGAS / SelfCheck prompt templates. This is **unconfirmed**: the API `usage.prompt_tokens_details.cached_tokens` was 0 in the logged calls.
- The ledger therefore over-states cost and never under-states it.
- Inkling ($0.003) is not listed. It is either too small to show at cent precision or not yet posted.
- The billing page updates with a delay, so the latest calls (A7) may not be included yet.

**Rule:** the **billed** amount is the official spent figure. The token-based rows are the per-run breakdown and the basis for estimates.

¹ Reasoning-token logging was added to the runner on 2026-11-27, after pilots 1 and 2.

---

## B. Approved, not spent / 已核准但未支出

| # | Approved (date, by) | Stage | Provider / model | Dataset / manifest SHA-256 | Units × reruns | Context | Budget cap (USD) | Spent so far (USD) | Remaining (USD) | Executor / payer | Status / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 2026-11-27, Emma | Pilot — stage 1 | Fireworks `muse-spark-open` + `inkling-2-small` | calib_v1.1 manifest `fe67dfe1…d6b0ee` | 207 claims × **1** each (ragas) | per Gate 6 decision (one context only) | **3.10** | 0.00 | 3.10 | Jimmy / Jimmy (to be shared, §6a) | **Approved, not run.** Starts after Gate 6 (context frozen) and the other formal-pilot gates. This is run 1 of 3 of the protocol; it is not three completed runs. |

---

## C. Pending approval / 尚待核准的預估支出

| # | Stage | Provider / model | Units × reruns | Estimate (USD) | Basis | Notes |
|---|---|---|---|---|---|---|
| C1 | Pilot — stage 2 (a) | Muse + Inkling | 207 × 2 more each | 5.0 – 6.3 | A5/A6 per-claim cost | Completes the 3-rerun protocol for both models. Applies if stage-1 quality is tied. |
| C2 | Pilot — stage 2 (b) | the leading model only | 207 × 2 more | Muse 4.1 – 5.0 / Inkling 0.8 – 1.2 | same | Applies if stage-1 quality differs clearly. |
| C3 | Pilot — stage 2 (c) | Muse + Inkling, rule-selected subset (~35%) | ~72 × 2 more each | 1.7 – 2.2 | same | Subset flip rate only, not full-set Det-var. This is a design change and must be recorded. |
| C4 | Eval (formal) | Muse (if selected) | eval_40pos_v1.1: 209 claims × 3 | 6.3 – 7.6 | same | Separate approval; never mixed with pilot. If Inkling is selected, about 1.3 – 1.9. |
| C5 | Cost optimisation check | Muse + Inkling, `reasoning_effort=low` vs default | design to be proposed | TBD | A7 | This is a protocol change. Validation design is needed first. |

Ranges are cited_span ~ all_docs. Inkling is estimated at $0.002 – 0.003 per claim from its smoke-test reasoning volume, and will be replaced by measured cost after stage 1.

---

## Price sources

| Provider / model | Input $/1M | Output $/1M | Source, date read |
|---|---|---|---|
| Fireworks `glm-5p3-flash` | 0.15 | 0.50 | app.fireworks.ai/models, uncached, 2026-11-27 |
| Fireworks `muse-spark-open` | 1.25 | 4.25 | app.fireworks.ai/models, uncached, 2026-11-27 |
| Fireworks `inkling-2-small` | 0.30 | 1.00 | app.fireworks.ai/models, uncached, 2026-11-27 |
| NVIDIA NIM (hosted API) | 0 | 0 | no charge observed; terms UNKNOWN, 2026-11-27 |
| Groq `openai/gpt-oss-20b` | 0 (free tier) | 0 (free tier) | free tier, 200k tokens/day |

## Change log
| Date | Change | By |
|---|---|---|
| 2026-11-27 | Ledger created: A1–A7 back-filled from run logs, B1 approved by Emma, C1–C5 pending. | Jimmy |
| 2026-11-27 | Reconciled with the Fireworks billing page: billed total $0.29. A1 DeepSeek part replaced by the billed $0.02; cumulative totals updated. | Jimmy |
