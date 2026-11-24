# Frozen data — calibration subset `calib_v1`

Gold-labelled claims from **RAGTruth QA** (human span annotations, MIT License) for calibrating the
verdict thresholds `(lo, hi)` of each scorer (Block 6). **Frozen at git tag `calib-v1`: never edit these files.**
A segmentation-rule change produces `calib_v1.1` (same responses, re-derived claims); different responses produce `calib_v2`.

## Rebuild (byte-identical)
```bash
git clone https://github.com/ParticleMedia/RAGTruth ../RAGTruth   # commit recorded in manifest.json
python3 tools/build_calib_subset.py --data-dir ../RAGTruth/dataset
sha256sum data/frozen/calib_v1/claims.jsonl                       # compare with manifest.json
```

## Split, seed, sampling
- Pool: `task_type == QA`, `split == train` (test is held out for the final harness), `quality == good`,
  excluding the source_ids used in `examples/`.
- `SEED = 20261021`: pool sorted by integer id, then `random.Random(SEED).shuffle`, round-robin over the 6 generator
  models, at most one response per `source_id`.
- Oversampled (agreed with the Gate owner, 2026-11-24): hallucinated responses until ≥ 50 hallucinated claims,
  ~1/3 of responses fully faithful, ≥ 150 claims total. See `STATS.md` for the counts and the prevalence-correction weights.

## Files (`calib_v1/`)
| file | layer | content |
|---|---|---|
| `responses.jsonl` | **frozen** | the selected RAGTruth response records, verbatim |
| `claims.jsonl` | derived | one row per claim (see schema below), recomputable from `responses.jsonl` |
| `bundles/ragtruth-{id}.json` | derived | AnswerBundle v1.1 per response (`schemas/answer_bundle.schema.json`) |
| `STATS.md` | derived | full-split + subset statistics, prevalence correction |
| `manifest.json` | — | seed, filters, input/code/output sha256 |

## `claims.jsonl` schema
| field | meaning |
|---|---|
| `claim_id` | `ragtruth-{response_id}-c{k}` |
| `response_id`, `source_id`, `model` | RAGTruth identifiers |
| `claim_text`, `char_start`, `char_end` | claim; invariant `response[char_start:char_end] == claim_text` |
| `excluded`, `exclude_reason` | `true` + `meta`/`short` for pieces that are not claims; **calibrate on `excluded == false` only** |
| `gold_verdict` | `PASS` (no human span overlap) / `UNSOURCED` (overlaps Baseless-Info spans only) / `FAIL` (overlaps a Conflict span) |
| `y_true` | 1 if the claim overlaps any human hallucination span, else 0 |
| `label_types`, `overlap_ratio` | overlapping RAGTruth label types; fraction of the claim's characters covered |

Claims come from `tools/segment.py` (rules v1.0, tests in `tools/test_segment.py`).
Gold for **any** claim span — e.g. from `blocks/segmenter.py` — can be computed with
`gold_for({"char_start": s, "char_end": e}, dedupe_labels(response["labels"]))` from `tools/build_calib_subset.py`,
using offsets into the response text.
