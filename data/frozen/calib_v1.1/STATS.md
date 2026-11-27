# Calibration subset calib_v1.1 — stats

Same 32 responses as calib_v1 (FROZEN layer reused, sha256-checked); claims and gold re-derived with
gold rules v1.1 (tools/gold_rules.py). Segmenter rules v1.0, MIN_WORDS=4.
Seed and sampling are those of calib_v1 (SEED=20261021, oversampled, split=train).

| | calib_v1 (rules v1.0) | calib_v1.1 (rules v1.1) |
|---|---|---|
| responses | 32 | 32 |
| claims (excluded=false) | 207 | 207 |
| gold PASS | 157 | 158 |
| gold FAIL (Conflict) | 10 | 10 |
| gold UNSOURCED (Baseless) | 40 | 39 |
| hallucinated-claim rate | 24.2% | 23.7% |

Changed labels: see LABEL_CHANGES.md.

## Prevalence correction

    w1 = pi / 0.2367        w0 = (1 - pi) / (1 - 0.2367)
    precision_pi = w1*TP / (w1*TP + w0*FP)      recall is unchanged

| target prevalence pi | value | w1 (y_true=1) | w0 (y_true=0) |
|---|---|---|---|
| QA train (natural, v1.1) | 10.8% | 0.4556 | 1.1688 |
| QA test (natural, v1.1) | 6.8% | 0.2857 | 1.2215 |

Thresholds calibrated on calib_v1 must not be reused for calib_v1.1; recalibrate and bind to this manifest.
