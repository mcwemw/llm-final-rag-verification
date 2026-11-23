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
- **Concat format / separator.** RAGTruth stores the context as one string:
  ```
  "passage 1:<text 1>\n\npassage 2:<text 2>\n\npassage 3:<text 3>\n\n"
  ```
  i.e. each passage is preceded by the header `passage N:` (10 chars for N < 10) and
  passages are separated by `\n\n` (2 chars). Citation spans cover only the passage text
  (headers and surrounding whitespace excluded). So the gap between consecutive spans is
  `"\n\npassage N:"` = **12 chars**, e.g. `answer_bundle_01`: p1 ends at 644, p2 starts at 656
  (`passages[644:656] == "\n\npassage 2:"`), p3 starts at 1204 = 1192 + 12.
  Verified on all 989 RAGTruth QA contexts: separator is always `\n\n`.
- `schema_version` is set to `"1.1"` in every bundle (optional field in the schema, so older
  v1.0 bundles without it still validate).
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
