# Data — RAGTruth

This repo does **not** vendor the corpus (~36 MB). It is the public
RAGTruth hallucination dataset from
[ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth).

Fetch the two files into this directory:

```bash
BASE=https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset
curl -L "$BASE/source_info.jsonl" -o data/source_info.jsonl
curl -L "$BASE/response.jsonl"    -o data/response.jsonl
```

Expected: `source_info.jsonl` ≈ 15 MB, `response.jsonl` ≈ 21 MB.

## Schema (used by `loader.py`)

`source_info.jsonl` — one grounding context per line:
`{source_id, task_type (QA|Summary|Data2txt), source, source_info, prompt}`.
For QA, `source_info = {question, passages}`; `passages` is the retrieved
context (the real RAG setting).

`response.jsonl` — one model answer per line:
`{id, source_id, model, temperature, labels, split (train|test), quality, response}`.
`labels` is the list of human-annotated hallucination spans; **empty list =
faithful**, non-empty = hallucinated. We use this at the response level:
`y_true = 1 if labels else 0`.
