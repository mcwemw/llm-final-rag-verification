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
| A1 | 2026-11-26 | Smoke | Fireworks `accounts/fireworks/models/glm-5p3-flash` | calib_v1 (1 response) | 6 claims × 1 (ragas) | cited_span | not logged | not logged | not logged | ≈0.0044 **(est.)** | 0.0044 | Jimmy / Jimmy | formula in notes | Spent, to be shared (§6a). Judge smoke test run before per-call logging existed. **Estimate** = 6 claims × pilot-2 RAGAS cost per claim-run = 6 × ($0.0332 / 45) = $0.0044. The $0.0332 is the actual RAGAS cost of 15 claims × 3 reruns in A3 (run `20261126T184015Z_calib_v1_fireworks`, `summary.json` → `scorers.ragas-llm-judge.cost_usd_actual`). Included in the billed GLM total. The est. tag stays until final reconciliation. |
| A2 | 2026-11-26 18:26 | Pilot (format) | Fireworks `accounts/fireworks/models/glm-5p3-flash` | calib_v1 · run-dataset `04f2f570…cab3b3` | 3 responses / 15 claims × 3 (presence, ragas, selfcheck) | cited_span | 93,926 | 54,766 | not recorded¹ | 0.0415 | 0.0459 | Jimmy / Jimmy | `runs/pilot_fireworks_glm/scores.jsonl`, run `20261126T182622Z_calib_v1_fireworks` | Spent, to be shared (§6a). Pilot 1. 2/45 RAGAS claims were truncated, which led to max_tokens=4096. GLM has since been excluded, so this is a format check, not a result. |
| A3 | 2026-11-26 18:40 | Pilot (format) | Fireworks `accounts/fireworks/models/glm-5p3-flash` | calib_v1 · run-dataset `04f2f570…cab3b3` | 3 responses / 15 claims × 3 (presence, ragas, selfcheck) | cited_span | 86,307 | 47,270 | not recorded¹ | 0.0366 | 0.0825 | Jimmy / Jimmy | `runs/pilot2_fireworks_glm/scores.jsonl`, run `20261126T184015Z_calib_v1_fireworks` | Spent, to be shared (§6a). Pilot 2. Its 9 exports are now format fixtures in `exp/fixtures/` (PR #11). |
| A4 | 2026-11-27 | Smoke | Fireworks `muse-spark-open`, `inkling-2-small` | — | 4 calls (JSON / plain × 2 models) | — | 382 | 813 | 761 | 0.0034 | 0.0859 | Jimmy / Jimmy | call responses (usage) in session log; Fireworks billing page | Spent, to be shared (§6a). Gate 1 connection and JSON-mode smoke test. Muse with max_tokens=400 spent all tokens on reasoning and returned empty content. |
| A5 | 2026-11-27 10:16 | Pilot (single-answer cost test) | Fireworks `accounts/fireworks/models/muse-spark-open` | calib_v1.1 manifest `fe67dfe1…d6b0ee` · run-dataset `9681014d…71cb0f7` | 1 response / 9 claims × 1 (ragas) | cited_span | 18,065 | 15,869 | 13,714 | 0.0900 | 0.1759 | Jimmy / Jimmy | `runs/t_muse/scores.jsonl`, `judge_calls.jsonl`, run `20261127T101640Z_calib_v1.1_fireworks` | Spent, to be shared (§6a). Measured per-claim cost: $0.0100. |
| A6 | 2026-11-27 10:19 | Pilot (single-answer cost test) | Fireworks `accounts/fireworks/models/muse-spark-open` | calib_v1.1 manifest `fe67dfe1…d6b0ee` · run-dataset `9681014d…71cb0f7` | 1 response / 9 claims × 1 (ragas) | all_docs | 20,629 | 19,722 | 17,338 | 0.1096 | 0.2855 | Jimmy / Jimmy | `runs/t_muse_all/scores.jsonl`, `judge_calls.jsonl`, run `20261127T101936Z_calib_v1.1_fireworks` | Spent, to be shared (§6a). Context comparison on the same answer. Rows include 2 claims re-run after a local timeout (resume). |
| A7 | 2026-11-27 | Smoke (cost optimisation) | Fireworks `muse-spark-open`, `inkling-2-small` | — | 6 calls (`reasoning_effort` default / low / medium × 2 models) | — | 1,125 | 7,273 | 6,411 | 0.0228 | 0.3083 | Jimmy / Jimmy | call responses (usage) in session log; Fireworks billing page | Spent, to be shared (§6a). `reasoning_effort=low` cuts reasoning tokens: Muse −69%, Inkling −65%. Using it would be a protocol change and is not approved. |
| — | 2026-11-26 – 27 | Pilot / Smoke | Groq `openai/gpt-oss-20b` (free tier) | calib_v1 | 199 claims × up to 3 | cited_span | — | — | — | **$0 (free tier, not billed)** | 0.3083 | Jimmy / — | Groq free tier (200k tokens/day), no payment method attached | Preliminary runs only, not results. The runner logged a nominal $0.0213 at third-party prices, but the free tier does not charge. |
| — | 2026-11-27 – | Smoke / Pilot | NVIDIA NIM `mistral-large-4-1t-a52b`, `nemotron-3-ultra-550b-a55b` | calib_v1.1 manifest `fe67dfe1…d6b0ee` | Nemotron: 207 claims × 3 × 2 contexts (running); Mistral: smoke only | cited_span, all_docs | 430,310 | 317,658 | 276,011 | **$0 observed (not billing-confirmed)** | 0.3083 | Jimmy / — | usage snapshot below (`runs/pilot_nemotron_*/scores.jsonl`, `judge_calls.jsonl`) | No charge seen so far. NIM has no billing page that confirms $0. Free-tier terms, RPM and data retention are **UNKNOWN**. Free is not assumed to be permanent or guaranteed. |

**NIM usage snapshot (cut-off 2026-11-27 12:18 UTC).** Usage is still growing while the pilot runs.

| Run | Rows (claim × rerun) | LLM calls | HTTP 200 / 429 | Input tok | Output tok | Reasoning tok | Last call (UTC) |
|---|---|---|---|---|---|---|---|
| `pilot_nemotron_cited_span` | 169 / 621 | 326 | 326 / 142 | 281,794 | 207,948 | 179,776 | 12:18:23 |
| `pilot_nemotron_all_docs` | 78 / 621 | 152 | 152 / 39 | 148,516 | 109,710 | 96,235 | 11:24:52 (paused, resumes after cited_span) |
| Manual smoke calls (Llama 410; Mistral / Nemotron / Nemotron-VL JSON tests; latency checks) | — | ≈7 | — | not logged | not logged | not logged | 2026-11-27 |

**Total spent (A): two figures, kept apart.**

| | Amount (USD) | What it is |
|---|---|---|
| **Provider-billed spend** | **0.27** | Fireworks billing snapshot (Spend, November, read 2026-11-27, cent-rounded): Muse 0.20 + GLM 0.07. This is the **spent figure**. |
| Token × list-price reconstructed cost | ≈ 0.308 | Sum of the A rows. It includes the A1 estimate ($0.0044, est.) and Inkling ($0.0031, see below). |
| **Reconciliation variance** | **≈ 0.038** | Reconstructed minus billed. **Cause not yet established**. The input side is billed at about 40% of list (a cached-prefix discount is suspected but unconfirmed); see the reconciliation table. |

- **No double counting.** A1 (≈$0.0044, est.) and every other Fireworks row are **already included** in the $0.27 provider-billed spend. Never add them to it.
- **Inkling** (`inkling-2-small`, A4 + A7, ≈$0.0031 reconstructed) has **not appeared on the billing page yet**. It is kept in the rows as usage and is not billing-confirmed.
- No estimated amount is counted as billed. Final reconciliation is **PENDING** (see below).

### Billing reconciliation (Fireworks Billing & Usage, Spend, snapshot read 2026-11-27): status **PENDING**

| Model | Billed: input (uncached) / output / total | Ledger (tokens × list price): input / output / total | Rows |
|---|---|---|---|
| Muse Spark Open | $0.02 / $0.18 / **$0.20** | $0.050 / $0.173 / $0.223 | A4, A5, A6, A7 |
| GLM 5.3 Flash | $0.01 / $0.05 / **$0.07** | $0.027 / $0.051 / ≈$0.082 | A1, A2, A3 |
| Inkling 2 Small | **not yet on billing page** | $0.0031 | A4, A7 |
| **Total** | **$0.27** (provider-billed) | $0.308 (reconstructed) | variance ≈ $0.038, cause not yet established |

**Findings.**
- **Output agrees** with the token-based figure: Muse $0.18 vs $0.173, GLM $0.05 vs $0.051.
- **Input is billed at about 40% of list.** The page shows only "Input (uncached)". A likely cause is a cached-prefix discount on the repeated RAGAS / SelfCheck prompt templates. This is **unconfirmed**: the API `usage.prompt_tokens_details.cached_tokens` was 0 in the logged calls.
- The ledger therefore over-states cost and never under-states it.
- Inkling ($0.003) is not listed. It is either too small to show at cent precision or not yet posted.
- The billing page updates with a delay, so the latest calls (A7) may not be included yet.

**Status: PENDING.** A snapshot was compared at cent precision, but the reconciliation is not complete:
1. Inkling ($0.003) has not appeared on the billing page.
2. The latest calls (A7) may not be posted yet.
3. The page rounds to cents, so per-row amounts below $0.01 cannot be confirmed.

Next check: after the billing page has posted all 2026-11-27 usage, then again at month-end. Mark the reconciliation DONE only when every Fireworks model with usage appears and the totals agree within rounding.

**Rule:** the amount on the billing page is the spent figure. The token-based rows are the per-run breakdown and the basis for estimates. Estimates are never recorded as billed.

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

## Per-call breakdown for A4 and A7

These are the API `usage` fields returned by each call. The cost is (input × input price + output × output price) / 1e6 at the list prices below. Each row can be recomputed independently. Reasoning tokens are part of output.

| Row | Call | Model | Input tok | Output tok | Reasoning tok | Cost (USD) |
|---|---|---|---|---|---|---|
| A4 | `response_format=json_object`, max_tokens 400 | muse-spark-open | 159 | 400 | 397 | 0.001899 |
| A4 | plain, max_tokens 400 | muse-spark-open | 159 | 255 | 238 | 0.001283 |
| A4 | `response_format=json_object`, max_tokens 400 | inkling-2-small | 32 | 81 | 65 | 0.000091 |
| A4 | plain, max_tokens 400 | inkling-2-small | 32 | 77 | 61 | 0.000087 |
| A4 | subtotal | muse-spark-open | 318 | 655 | 635 | 0.003182 |
| A4 | subtotal | inkling-2-small | 64 | 158 | 126 | 0.000178 |
| **A4** | **total** | | **382** | **813** | **761** | **0.003360** |
| A7 | `reasoning_effort` default | muse-spark-open | 239 | 2,108 | 1,961 | 0.009258 |
| A7 | `reasoning_effort=low` | muse-spark-open | 239 | 739 | 610 | 0.003440 |
| A7 | `reasoning_effort=medium` | muse-spark-open | 239 | 1,604 | 1,460 | 0.007116 |
| A7 | `reasoning_effort` default | inkling-2-small | 136 | 1,302 | 1,156 | 0.001343 |
| A7 | `reasoning_effort=low` | inkling-2-small | 136 | 558 | 410 | 0.000599 |
| A7 | `reasoning_effort=medium` | inkling-2-small | 136 | 962 | 814 | 0.001003 |
| A7 | subtotal | muse-spark-open | 717 | 4,451 | 4,031 | 0.019814 |
| A7 | subtotal | inkling-2-small | 408 | 2,822 | 2,380 | 0.002944 |
| **A7** | **total** | | **1,125** | **7,273** | **6,411** | **0.022758** |

All A7 calls used T=0, top_p=1, max_tokens 4096 and `json_object`. The A7 prompt is a 2-statement judging prompt over a 2-sentence context, the same for all six calls.

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
| 2026-11-27 | Compared with a Fireworks billing snapshot (cent-rounded); cumulative totals updated. | Jimmy |
| 2026-11-27 | Review fixes (Emma, PR #12): A1 estimate formula and source; per-call breakdown for A4/A7; NIM changed to "$0 observed (not billing-confirmed)" with a usage snapshot; billing reconciliation kept **PENDING**. | Jimmy |
| 2026-11-27 | Second review (Emma): totals split into provider-billed / reconstructed / variance ≈$0.038; A1 and all Fireworks rows marked as included in $0.29 (no double counting); Inkling marked not yet on the billing page; per-model subtotals added for A4/A7. B1 and C unchanged. | Jimmy |
| 2026-11-27 | Scope decision (team lead): A1 limited to the GLM judge smoke test; provider-billed spend recorded as $0.27 (Muse + GLM); reconstructed ≈$0.308; cumulative totals updated. | Jimmy |
