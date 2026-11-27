"""Program 2 (local port) — retrieve, answer with citations, export AnswerBundle (Blocks 1-2).

Local port of RAG_02_使用篇.ipynb (yenlung/Generative-AI @655a786): same retrieval, model
(openai/gpt-oss-120b on Groq), temperature, system prompt and citation_note(). The system prompt,
build_context/build_messages, safe_api_error and citation_note are the course's own code, loaded
from the downloaded notebook (rag_base.course_assets), not copied. Changes vs. the notebook:
  * Colab upload / Colab Secrets  -> --kb path / GROQ_API_KEY (env or .env)
  * aisuite "groq:openai/gpt-oss-120b" -> the groq SDK it wraps (same model, same params)
  * Gradio UI dropped; AnswerBundle v1.1 export added for the verification layer (Block 2).

Usage (repo root):
    python -m rag_base.course_assets                                   # once
    python -m rag_base.program2_answer --question-set --retrieve-only  # no API key, hit@k
    python -m rag_base.program2_answer -q "工讀生也有特休嗎？" --name tekyu
    python -m rag_base.program2_answer --question-set                  # 19 Groq calls
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rag_base import course_assets
from rag_base.rag_kb import KnowledgeBase

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCHEMA_PATH = REPO / "schemas" / "answer_bundle.schema.json"
OUT_DIR = HERE / "out"  # gitignored

GROQ_MAX_RETRIES = 6
LLM_MODEL = "openai/gpt-oss-120b"   # notebook: "groq:openai/gpt-oss-120b" via aisuite
DEFAULT_TOP_K = 3
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 4096

def _key_from_env_files(name: str) -> str:
    for env in (REPO / ".env", REPO.parent / ".env"):
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip().removeprefix("export ")
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def make_client():
    key = os.environ.get("GROQ_API_KEY", "").strip() or _key_from_env_files("GROQ_API_KEY")
    if not key:
        raise SystemExit(
            "找不到 GROQ_API_KEY。請先 `export GROQ_API_KEY=gsk_...`，或寫進 repo 根目錄或上一層的 .env"
            "（.env 不要 commit）。只想測檢索可加 --retrieve-only。"
        )
    from groq import Groq
    # Free-tier Groq returns 429 when several questions run back-to-back (per-minute limit).
    # The SDK retries with exponential backoff and honours Retry-After; 1 retry was not enough
    # (live demo 2026-11-27: 4/19 questions lost). Retrying does not change the model call.
    return Groq(api_key=key, timeout=90.0, max_retries=GROQ_MAX_RETRIES)


def call_llm(client, messages: list[dict], safe_api_error) -> tuple[str, dict]:
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL, messages=messages,
            temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
        )
        choice = response.choices[0]
        answer = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
    except Exception as exc:
        raise RuntimeError(safe_api_error(exc)) from None
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("模型沒有回傳可顯示的文字；請稍後再試，或檢查輸出 token 額度。")
    usage = getattr(response, "usage", None)
    info = {
        "finish_reason": finish_reason,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    # The notebook appends a truncation warning to the answer; we keep the answer clean
    # (it is what gets verified) and record finish_reason in the .meta.json instead.
    return answer.strip(), info


# ── AnswerBundle export (Block 2 contract, schema v1.1) ──────────────────────
_DOC_ID = re.compile(r"doc_[A-Za-z0-9_-]+")


def cited_ids_in_order(answer: str) -> list[str]:
    """Every doc id mentioned anywhere in the answer, in first-appearance order.

    Broader than the course citation_note() regex (half-width [doc_xxx] only): gpt-oss-120b also writes
    【doc_016】, 〔doc_006〕, （doc_011）, [doc_003, doc_010] ... so we match the id itself, not the brackets.
    """
    seen: list[str] = []
    for doc_id in _DOC_ID.findall(answer):
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def to_answer_bundle(question: str, answer: str, results: list[dict]) -> tuple[dict, list[str]]:
    """Returns (bundle, cited ids that were NOT retrieved).

    Program 2 sends whole documents (one vector per document, no chunking), so a citation
    covers the full document text: span = [0, len(text)).
    """
    by_id = {r["doc_id"]: r for r in results}
    cited = cited_ids_in_order(answer)
    citations = [
        {"doc_id": d, "span_start": 0, "span_end": len(by_id[d]["text"]), "text": by_id[d]["text"]}
        for d in cited if d in by_id
    ]
    bundle = {
        "schema_version": "1.1",
        "question": question,
        "answer": answer,
        "citations": citations,
        "retrieved": [{"doc_id": r["doc_id"], "text": r["text"]} for r in results],
    }
    return bundle, [d for d in cited if d not in by_id]


def validate_bundle(bundle: dict, kb: KnowledgeBase) -> None:
    """JSON Schema + the three Block 2 invariants from spec-sheet 02."""
    if SCHEMA_PATH.exists():
        import jsonschema
        jsonschema.validate(bundle, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    else:
        print(f"  ⚠ 找不到 schema（{SCHEMA_PATH}），略過 JSON Schema 驗證")
    for c in bundle["citations"]:
        doc = kb.documents_by_id.get(c["doc_id"])
        assert doc is not None, f"doc_id 不在語料庫：{c['doc_id']}"
        assert 0 <= c["span_start"] <= c["span_end"] <= len(doc["text"]), f"span 超出範圍：{c['doc_id']}"
        assert doc["text"][c["span_start"]:c["span_end"]] == c["text"], f"引用文字被竄改：{c['doc_id']}"


# ── question set (問題集.csv inside rag-data.zip) ─────────────────────────────
def load_question_set(zip_path: Path) -> list[dict]:
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read("問題集.csv").decode("utf-8-sig")
    return [
        {"question": row["測試問題"], "expected_file": row["預期相關檔案"], "type": row["類型"]}
        for row in csv.DictReader(io.StringIO(raw))
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--course-dir", default=str(course_assets.DEFAULT_DIR), help="downloaded course materials")
    ap.add_argument("--kb", default=None,
                    help="Program 1 匯出的知識庫（zip 或資料夾）；預設用老師預建的 vector_embedding.zip")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-q", "--question", help="單一問題")
    g.add_argument("--question-set", action="store_true", help="跑 rag-data.zip 內的 問題集.csv（19 題）")
    ap.add_argument("--name", default=None, help="單一問題時的輸出檔名（預設用時間戳）")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--retrieve-only", action="store_true", help="只做檢索，不呼叫 Groq、不輸出 bundle")
    ap.add_argument("--out", default=str(OUT_DIR / "bundles"))
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = ap.parse_args()

    course = course_assets.load(Path(args.course_dir))
    args.kb = args.kb or str(Path(args.course_dir) / "vector_embedding.zip")
    kb = KnowledgeBase(args.kb, device=args.device)
    print(f"知識庫：{Path(args.kb).name}／{kb.index.ntotal} 篇／"
          f"模型 {kb.config['model_id']}@{kb.config['model_revision'][:7]}／{kb.device}")
    file_to_id = {d["metadata"].get("filename"): d["doc_id"] for d in kb.documents}

    if args.question_set:
        items = load_question_set(Path(args.course_dir) / "rag-data.zip")
        for i, it in enumerate(items, start=1):
            it["name"] = f"q{i:02d}"
    else:
        name = args.name or datetime.now(ZoneInfo("Asia/Taipei")).strftime("q_%Y%m%d_%H%M%S")
        items = [{"question": args.question, "expected_file": None, "type": None, "name": name}]

    client = None if args.retrieve_only else make_client()
    out_dir = Path(args.out)
    hits = total_in_kb = 0
    failed: list[str] = []
    for it in items:
        results = kb.search(it["question"], args.top_k)
        expected_id = file_to_id.get(it["expected_file"])
        mark = ""
        if expected_id:
            total_in_kb += 1
            hit = any(r["doc_id"] == expected_id for r in results)
            hits += hit
            mark = "  ✓" if hit else f"  ✗（預期 {expected_id}）"
        print(f"\n[{it['name']}] {it['question']}{mark}")
        for r in results:
            print(f"   {r['rank']}. [{r['doc_id']}] {r['metadata'].get('title')}  cos={r['score']:.4f}")
        if client is None:
            continue

        try:
            answer, info = call_llm(client, course.build_messages(it["question"], results),
                                    course.safe_api_error)
        except RuntimeError as exc:
            print(f"   ✗ {exc}")
            failed.append(it["name"])
            continue
        note = course.citation_note(answer, results)
        bundle, unknown = to_answer_bundle(it["question"], answer, results)
        validate_bundle(bundle, kb)
        meta = {
            "question_type": it["type"],
            "expected_doc_id": expected_id,
            "llm_model": LLM_MODEL,
            "course_commit": course.commit,
            "temperature": TEMPERATURE,
            "top_k": args.top_k,
            "kb": Path(args.kb).name,
            "embedding_model": f"{kb.config['model_id']}@{kb.config['model_revision']}",
            "retrieval": [{"rank": r["rank"], "doc_id": r["doc_id"], "score": r["score"]} for r in results],
            "cited_not_retrieved": unknown,
            "citation_note": note,
            "created_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
            **info,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{it['name']}.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / f"{it['name']}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"   → {it['name']}.json  引用 {[c['doc_id'] for c in bundle['citations']]}"
              f"{'  ⚠ 引用了未取回的 ' + str(unknown) if unknown else ''}"
              f"{'  ⚠ 輸出被截斷' if info['finish_reason'] == 'length' else ''}")
        print(f"   {note}")

    if total_in_kb:
        print(f"\n檢索命中率 hit@{args.top_k}：{hits}/{total_in_kb}（不含「應拒絕臆測」題）")
    if client is not None:
        print(f"輸出資料夾：{out_dir}")
        done = len(items) - len(failed)
        print(f"完成 {done}/{len(items)} 題" + (f"；失敗：{', '.join(failed)}" if failed else ""))
    raise SystemExit(exit_code(failed))


def exit_code(failed: list[str]) -> int:
    """Non-zero when any question got no answer, so a partial run never looks complete."""
    return 1 if failed else 0


if __name__ == "__main__":
    main()
