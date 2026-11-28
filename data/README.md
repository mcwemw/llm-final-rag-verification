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

## Preprocessing note (RAGTruth → AnswerBundle)
RAGTruth responses have no inline citation markers, so **every retrieved passage is treated
as cited** and claims are bound to sources with **`nearest` alignment**. Span offsets index the
full `source_info.passages` string; see [`../examples/README.md`](../examples/README.md).

## Corpus text, licensing and hydration (2026-11-28)
**No corpus text is committed to this public repo.**
- RAGTruth's LICENSE is MIT and names "the Software", with copyright 2023 Particle Media. It does not say the data files are covered. Source: https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE, read 2026-11-28.
- The RAGTruth QA passages come from MS MARCO (`"source": "MARCO"`). MS MARCO's terms say the data is "intended for non-commercial research purposes only … without extending any license or other intellectual property rights". They neither grant nor explicitly forbid redistribution. Source: https://microsoft.github.io/msmarco/ ("Terms and Conditions"), read 2026-11-28.
- Not yet determined: whether MIT covers the RAGTruth data files, and the upstream terms of the CNN/DM and Yelp sources (not used here: we use QA only).

**How frozen sets are stored.** Every frozen set (`calib_v1`, `calib_v1.1`, `eval_40pos_v1.1`) keeps only these files:
- `manifest.json`: unchanged. It holds the sha256 of every full file and is the acceptance target.
- `STATS.md`
- `storage.json`
- `refs/`: ids, offsets and labels only. There is no question, answer, passage text, claim text or annotator note.

Rebuild the full files locally into `data/hydrated/` (gitignored) after fetching the two RAGTruth files above:
```bash
python3 tools/hydrate.py --all              # or one set: python3 tools/hydrate.py calib_v1.1
```
Use the hydrated copies as inputs, e.g. `--claims data/hydrated/calib_v1.1/claims.jsonl` (the hydrated
folder also holds `manifest.json`, so the manifest check of the frozen-claims loader still applies).
`calib_v1.1/LABEL_CHANGES.md` (it quotes claim text) is also rebuilt locally; its scope is the whole QA pool.
It fails closed and writes nothing in any of these cases:
- a corpus file is missing, or its sha256 differs from the manifest;
- a reference file was modified;
- an id or offset does not match the corpus;
- any rebuilt file is not byte-identical to the manifest;
- the output path is tracked or not gitignored.

Migration evidence (hashes only) is in `data/migration_evidence/<set>/`.

**Builders.** `tools/build_calib_subset.py`, `tools/build_calib_v1_1.py` and `tools/build_eval_set.py` are
not modified (their sha256 are recorded in the manifests). They read full files under `data/frozen/`, so
they no longer run on a fresh checkout. Reproduction is `tools/hydrate.py`, which checks every rebuilt file
byte-for-byte against the manifest written by the original build.

**Not migrated yet:** `examples/` (5 example bundles + gold). The tests of several open branches read
them, so they move in a later PR, after those branches merge.
