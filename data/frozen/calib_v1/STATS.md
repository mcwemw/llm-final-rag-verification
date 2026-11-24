# Calibration subset calib_v1 — stats

```
RAGTruth QA split (quality=good, segmenter rules v1.0)
train  #responses: 4,892   #claims (span->claim): 32,628
       gold: PASS 29,104 / WARN 0* / FAIL 488 / UNSOURCED 3,036   (hallucinated 10.8%)
test   #responses: 875   #claims (span->claim): 5,264
       gold: PASS 4,907 / WARN 0* / FAIL 32 / UNSOURCED 325   (hallucinated 6.8%)
  *WARN is a gate output class; gold has only supported / not supported / no source
Calibration subset (frozen, from train, SEED=20261021): 207 claims from 32 responses
       gold: PASS 157 / WARN 0* / FAIL 10 / UNSOURCED 40
  double-annotation agreement (kappa): not done yet
```

## Calibration subset details

RAGTruth QA, split=train, quality=good, SEED=20261021, sampling: oversampled (>= 50 hallucinated claims, ~33% of responses faithful)
Segmenter rules v1.0 (tools/segment.py), MIN_WORDS=4

| | count |
|---|---|
| responses | 32 (hallucinated 24 / faithful 8) |
| claims (calibration, excluded=false) | 207 |
| gold PASS | 157 |
| gold FAIL (Conflict) | 10 |
| gold UNSOURCED (Baseless) | 40 |
| gold WARN | 0 (WARN is a gate output, not a gold class) |
| hallucinated-claim rate, this subset | 24.2% |
| hallucinated-claim rate, natural (whole pool, same segmenter) | 10.8% |
| excluded pieces (meta / short) | 13 / 0 |
| excluded pieces overlapping a human span (lost by segmenter) | 0 |

Per generator model (responses): gpt-3.5-turbo-0613 6, gpt-4-0613 6, llama-2-13b-chat 5, llama-2-70b-chat 5, llama-2-7b-chat 5, mistral-7B-instruct 5

Gold mapping: no span overlap -> PASS; Baseless only -> UNSOURCED; any Conflict -> FAIL. y_true = 1 if any overlap.
Note: thresholds tuned at the subset rate shift slightly vs the natural rate; final metrics are reported on `test`.

## Prevalence correction

The subset is oversampled (24.2% hallucinated). To estimate metrics at a natural prevalence pi,
weight each calibration claim by w1 (y_true=1) or w0 (y_true=0) when counting TP/FP/FN/TN:

    w1 = pi / 0.2415        w0 = (1 - pi) / (1 - 0.2415)
    precision_pi = w1*TP / (w1*TP + w0*FP)      recall is unchanged (only positives in TP and FN)

| target prevalence pi | value | w1 (y_true=1) | w0 (y_true=0) |
|---|---|---|---|
| QA train (natural) | 10.8% | 0.4478 | 1.1759 |
| QA test (natural) | 6.8% | 0.2808 | 1.2291 |

Recall does not depend on prevalence, so a fixed-recall threshold (e.g. recall = 0.8 on hallucinated claims)
transfers from this subset to natural data without correction; max-F1 thresholds should use the weights above.
