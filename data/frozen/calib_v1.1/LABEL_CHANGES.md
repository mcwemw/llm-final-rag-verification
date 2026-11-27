# Gold rules v1.0 → v1.1: label-change report

**Rule change.** A human span overlaps a claim only if the overlapping characters contain at least one
letter or digit (`str.isalnum`). Pure punctuation or whitespace overlap is no longer content overlap.
Segmentation is unchanged (tools/segment.py rules v1.0); only the gold mapping is versioned.

**Scope.** Every non-excluded claim of every RAGTruth QA response with quality=good (train + test).
The columns compare gold verdict, y_true and label types between the two rules.

| | v1.0 positives | v1.1 positives | changed |
|---|---|---|---|
| train | 3524 | 3519 | 5 |
| test | 357 | 356 | 1 |

All 6 changed claims:

| claim_id | split | in | v1.0 → v1.1 | overlapping text | claim |
|---|---|---|---|---|---|
| `ragtruth-14293-c2` | test | test pool | UNSOURCED (y=1) → PASS (y=0) | '.' | It measures the amount of carbon dioxide and other greenhouse gases produced as a result o |
| `ragtruth-14594-c5` | train | train pool | UNSOURCED (y=1) → PASS (y=0) | '.' | However, there is no mention of specific time frame for healing. |
| `ragtruth-15286-c3` | train | train pool | UNSOURCED (y=1) → PASS (y=0) | '.' | those aged 31-50 require 1,800-2,200 calories, and those over 50 require 1,600-2,200 calor |
| `ragtruth-17277-c3` | train | train pool | UNSOURCED (y=1) → PASS (y=0) | '.' | Muscles produce movement by contracting and pulling on tendons, which in turn move the bon |
| `ragtruth-17279-c2` | train | **calib_v1 → calib_v1.1** | UNSOURCED (y=1) → PASS (y=0) | '.' | Skeletal muscles, which are the only voluntary muscle tissue in the human body, contract t |
| `ragtruth-17636-c4` | train | train pool | UNSOURCED (y=1) → PASS (y=0) | '.' | The earliest sunset is 5:48 PM on December 8 and the latest sunset is 5:57 PM on December  |

Response-level labels (`y_true = 1 if labels`) do not depend on this rule, so no response-level
label changes. eval_40pos is built with v1.1 from the start; see its own STATS.md.
