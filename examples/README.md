# AnswerBundle examples (Block 1 → Block 3)

5 example `AnswerBundle`s for end-to-end testing of the verification layer.
Each file validates against [`../schemas/answer_bundle.schema.json`](../schemas/answer_bundle.schema.json) **v1.1**
(includes the optional `retrieved` field).

| file | RAGTruth response id | generator model | human label |
|---|---|---|---|
| `answer_bundle_01.json` | 13772 | mistral-7B-instruct | faithful |
| `answer_bundle_02.json` | 14016 | gpt-4-0613 | faithful |
| `answer_bundle_03.json` | 13706 | mistral-7B-instruct | hallucinated (1 span) |
| `answer_bundle_04.json` | 13259 | llama-2-70b-chat | hallucinated (1 span) |
| `answer_bundle_05.json` | 14884 | llama-2-13b-chat | hallucinated (2 spans) |

Source: RAGTruth QA subset (MS MARCO questions), **train split only** (no test leakage), seed 0.

## Conventions (confirmed with the Gate owner)
- **One citation per retrieved passage.** RAGTruth answers carry no inline citation markers;
  the generator was told to answer *only* from these 3 passages, so every passage is treated as cited
  and claim↔source binding uses `nearest` alignment.
- `doc_id = "ragtruth-{source_id}-p{N}"` (N = passage number).
- **Span offsets are into the whole concatenated context, NOT per-passage.**
  `span_start` / `span_end` index the original RAGTruth `source_info.passages` string
  (all passages concatenated, including the `passage N:` headers), so they do **not** start at 0
  for each passage. Invariant: `passages[span_start:span_end] == text`.
- `retrieved` (schema v1.1) lists all retrieved passages D as `{doc_id, text}`.
  For RAGTruth it is identical to `citations` (every retrieved passage is cited); for the
  course Program 2 demo it will also contain retrieved-but-uncited passages.
- Human hallucination labels are **not** part of AnswerBundle (schema forbids extra fields).
  They are in `answer_bundle_gold.json`, keyed by file name; `start`/`end` are offsets into `answer`.

## Regenerate
```bash
# fetch data first (see ../data/README.md)
python3 scripts/make_answer_bundle_examples.py --n 5 --seed 0
```
