# Evaluation set eval_40pos_v1 — stats (CANDIDATE, not frozen)

RAGTruth QA, **split=test**, quality=good, SEED=20261127, segmenter rules v1.0 (MIN_WORDS=4).
Sampling is the calib_v1 procedure, imported from `tools/build_calib_subset.py` (one response per source_id,
round-robin over generator models, oversampled: >= 40 hallucinated claims,
~33% of responses fully faithful, stop at response boundaries).

| | count |
|---|---|
| responses | 32 (hallucinated 23 / faithful 9) |
| claims (scored, excluded=false) | 181 |
| **hallucinated claims (y_true=1)** | **40** |
| gold PASS | 141 |
| gold FAIL (Conflict) | 4 |
| gold UNSOURCED (Baseless) | 36 |
| hallucinated-claim rate, this set | 22.1% |
| hallucinated-claim rate, natural (whole pool, same segmenter) | 6.7% |
| hallucinated-response rate, this set / pool | 71.9% / 17.8% |
| excluded pieces (meta / short) | 17 / 3 |
| excluded pieces overlapping a human span (lost by segmenter) | 0 |
| positives with overlap_ratio < 0.3 (see POSITIVES_REVIEW.md) | 6 |

Hallucinated claims per generator model: gpt-3.5-turbo-0613 10, gpt-4-0613 1, llama-2-13b-chat 6, llama-2-70b-chat 7, llama-2-7b-chat 7, mistral-7B-instruct 9

## Positive definition

Gold comes only from RAGTruth's human span annotations, never from an LLM (same mapping as calib_v1):
a claim is hallucinated (y_true=1) if its character span overlaps any annotated span;
any Conflict span -> FAIL, Baseless-only -> UNSOURCED, no overlap -> PASS.
Every positive is listed with the overlapping human span and the annotator note in `POSITIVES_REVIEW.md`.

## Leakage checks (all asserted at build time)

| check | result |
|---|---|
| split of every response | test (calib_v1 is train) |
| response_id overlap with calib_v1 | 0 |
| source_id overlap with calib_v1 | 0 |
| question overlap with calib_v1 | 0 |
| document (passage) overlap with calib_v1 | 0 |
| document (passage) overlap with the whole train split | 0 |
| one response per source_id | True |

Documents are compared per passage after whitespace/case normalisation. Test sources dropped from the pool before sampling:

| source_id | reason |
|---|---|
| 15112 | 1 passage(s) also in train |
| 15317 | 1 passage(s) also in train |

## Strata and sampling weights

Unit drawn = response. Stratum = generator model x response label. `weight` = pool / sampled
(inverse inclusion fraction) for population estimates at the natural test distribution.

| model | response label | pool | sampled | inclusion | weight |
|---|---|---|---|---|---|
| gpt-3.5-turbo-0613 | faithful | 134 | 2 | 1.5% | 67.0 |
| gpt-3.5-turbo-0613 | hallucinated | 5 | 5 | 100.0% | 1.0 |
| gpt-4-0613 | faithful | 147 | 2 | 1.4% | 73.5 |
| gpt-4-0613 | hallucinated | 1 | 1 | 100.0% | 1.0 |
| llama-2-13b-chat | faithful | 113 | 2 | 1.8% | 56.5 |
| llama-2-13b-chat | hallucinated | 35 | 5 | 14.3% | 7.0 |
| llama-2-70b-chat | faithful | 114 | 1 | 0.9% | 114.0 |
| llama-2-70b-chat | hallucinated | 33 | 4 | 12.1% | 8.25 |
| llama-2-7b-chat | faithful | 96 | 1 | 1.0% | 96.0 |
| llama-2-7b-chat | hallucinated | 51 | 4 | 7.8% | 12.75 |
| mistral-7B-instruct | faithful | 105 | 1 | 0.9% | 105.0 |
| mistral-7B-instruct | hallucinated | 29 | 4 | 13.8% | 7.25 |

**In-sample vs population.** Report in-sample metrics as measured on this set (prevalence 22.1%).
For claim-level estimates at the natural test prevalence (6.7%), weight each claim by
w1 = 0.3033 (y_true=1) or w0 = 1.1976 (y_true=0):

    precision_pi = w1*TP / (w1*TP + w0*FP)      recall is unchanged

AUROC and recall do not depend on prevalence; precision, F1 and max-F1 thresholds do.
