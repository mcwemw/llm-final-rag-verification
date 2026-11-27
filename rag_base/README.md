# rag_base — course Program 1 / 2, local port (Blocks 1–2)

A local, command-line port of the course RAG system: Program 1 builds the knowledge base and Program 2 answers with citations. Program 2 also exports **AnswerBundle v1.1** for the verification layer.

It is used for the **Chinese end-to-end demo**. The main experiment runs on RAGTruth.

## Licensing: what is (not) in this repo
The course notebooks and corpus ([yenlung/Generative-AI](https://github.com/yenlung/Generative-AI)) have **no license**, so this repo does **not** redistribute them.

- Each user downloads them from the author's repository at a pinned commit (`655a786`), and they are checked against sha256 hashes.
- The course functions we use **unchanged** are read from the downloaded notebooks at run time ([course_assets.py](course_assets.py)) instead of being copied here: `SYSTEM_PROMPT`, `build_context`, `build_messages`, `safe_api_error`, `citation_note`, `extract_rag_text` and `extract_metadata`.
- The rest is our own local rewrite: KB loading and checks, retrieval, the Groq call, and AnswerBundle export.
- `rag_base/course/` (downloaded materials) and `rag_base/out/` (KB, bundles; they contain corpus text) are gitignored.

## Setup
These dependencies go on top of the repo's venv: `faiss-cpu==1.15.1`, `sentence-transformers`, `torchvision==0.29.2` (needed by EmbeddingGemma 2), `groq`, `jsonschema`.

```bash
python -m rag_base.course_assets          # download + verify the course materials (once)
```

## Usage (repo root)
```bash
# Program 1: build the KB from the course corpus -> rag_base/out/kb/ (+ vector_embedding.zip)
python -m rag_base.program1_build_index

# Program 2: retrieval only (no API key) on the course question set, reports hit@3
python -m rag_base.program2_answer --question-set --retrieve-only

# Program 2: answer with Groq openai/gpt-oss-120b -> rag_base/out/bundles/qXX.json (+ .meta.json)
python -m rag_base.program2_answer --question-set          # GROQ_API_KEY from env or .env
python -m rag_base.program2_answer -q "工讀生也有特休嗎？" --name tekyu
```
By default, Program 2 uses the course's prebuilt KB. Pass `--kb rag_base/out/kb/vector_embedding.zip` to use your own.

## Differences from the Colab notebooks (I/O only)
| | Colab | here |
|---|---|---|
| input files | `files.upload()` | `--course-dir`, `--data`, `--kb` |
| API key | Colab Secrets | `GROQ_API_KEY` (env or `.env`, never committed) |
| LLM call | `aisuite` `groq:openai/gpt-oss-120b` | the `groq` SDK it wraps; same model, T=0.2, max_tokens 4096 |
| UI | Gradio | CLI |
| output | rendered answer | AnswerBundle v1.1 + `.meta.json` (retrieval scores, tokens, finish_reason, course citation note) |

## AnswerBundle conversion
- Program 2 embeds **one vector per document** (no chunking), so the LLM sees whole documents. A citation therefore spans the whole document: `span_start=0`, `span_end=len(text)`.
- `citations`: every retrieved `doc_xxx` the answer mentions, in first-appearance order.
  - It matches the id itself, not the brackets, because `gpt-oss-120b` writes `【doc_016】`, `〔doc_006〕`, `（doc_011）`, `[doc_003, doc_010]` and so on.
- `retrieved`: all top-k documents.
- Cited ids that were not retrieved are fabricated citations. They go into `.meta.json` → `cited_not_retrieved`.
- Every bundle is checked before it is written: the JSON Schema plus the three invariants of spec 02 (doc_id exists, span in range, `text[span] == citation.text`).

## Verified (2026-11-27, RTX 4070)
- **Recorded live demo:** see [LIVE_DEMO_2026-11-27.md](LIVE_DEMO_2026-11-27.md). It ran from a fresh worktree and a fresh download, with every step exit=0:
  - the 4 course files match the pinned sha256;
  - Program 1 rebuilds a `documents.json` **identical** to the course's prebuilt KB;
  - retrieval hit@3 is 15/16 on both the prebuilt and the rebuilt KB;
  - Program 2 answered **19/19** questions with the rebuilt KB, and all bundles pass the schema and the spec-02 invariants.
- The record has the environment (Python / torch / transformers / GPU), the sha256 of every input and output, and the run log location.
- The miss in hit@3 is q01: the expected doc_001 is not in the top 3. The 3 "should refuse" questions are excluded.
- The functions loaded from the notebooks give output identical to the earlier copied port. This was checked on all 16 documents and on 14 real Program 2 answers.
- **Groq free tier:** Program 2 retries rate limits (429) up to 6 times with backoff. It exits non-zero and lists the failed questions if any question gets no answer. Before this fix, a run lost 4/19 questions and still exited 0.

## Findings for the report
- **The course's own presence-only check misses citations.** `citation_note()` only matches half-width `[doc_xxx]`, but `gpt-oss-120b` mostly writes full-width or other brackets (`【doc_xxx】`, `〔doc_xxx〕`, `（doc_xxx）`).
  - Recorded run (2026-11-27): 18/19 answers cite documents, and `citation_note()` reports "回答未附文件編號" (no citation attached) for **14 of the 18**. The 4 it recognises use half-width brackets.
  - An earlier run the same day: **17 of 19**.
  - Answers are sampled at T=0.2, so the count varies per run. The cause does not.
- Refusals can also mention doc ids. For example, q17 ("這家店的店長叫什麼名字？") is correctly refused, but it names `(doc_004)` and others as the documents that do *not* contain the answer. The bundle export treats any mention as a citation.
- The English Block 3/4 rules cannot segment or align these Chinese answers. See the `--lang zh` proposal in PR #4.
