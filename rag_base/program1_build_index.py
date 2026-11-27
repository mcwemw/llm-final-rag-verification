"""Program 1 (local port) — build the course RAG knowledge base (Block 1).

Local port of RAG_01_打造你的_RAG_系統.ipynb (yenlung/Generative-AI @655a786): same model,
templates, FAISS index and export format. extract_rag_text() / extract_metadata() are the
course's own functions, loaded from the downloaded notebook (rag_base.course_assets), not copied.

Usage (repo root):
    python -m rag_base.course_assets                      # download course materials once
    python -m rag_base.program1_build_index               # course corpus -> rag_base/out/kb/
    python -m rag_base.program1_build_index --data my.zip --out my_kb --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as package_metadata
import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import faiss
import numpy as np
from huggingface_hub import HfApi

from rag_base import course_assets
from rag_base.rag_kb import (
    DOCUMENT_TEMPLATE, KB_FILES, MODEL_CONFIG_KWARGS, MODEL_ID, QUERY_TEMPLATE,
    count_input_tokens, embed_question, load_model, pick_device,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"  # gitignored
PACKAGE_NAMES = ["torch", "numpy", "sentence-transformers", "transformers", "faiss-cpu", "huggingface-hub"]


# ── cell 5: unzip TXT only, with path-traversal guard ─────────────────────────
def unzip_txt(archive_path: Path, data_dir: Path) -> list[Path]:
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            m for m in archive.infolist()
            if not m.is_dir() and m.filename.lower().endswith(".txt")
            and not any(part.startswith((".", "__MACOSX")) for part in Path(m.filename).parts)
        ]
        if not members:
            raise ValueError("ZIP 中找不到 TXT；請確認使用第一版教材，或準備自己的 TXT。")
        destinations = set()
        for member in members:
            target = (data_dir / member.filename).resolve()
            if not target.is_relative_to(data_dir.resolve()):
                raise ValueError(f"ZIP 包含不適當的檔案路徑：{member.filename}")
            if target in destinations:
                raise ValueError(f"ZIP 有重複檔名：{member.filename}")
            destinations.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    return sorted(
        [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: p.relative_to(data_dir).as_posix(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--course-dir", default=str(course_assets.DEFAULT_DIR), help="downloaded course materials")
    ap.add_argument("--data", default=None, help="教材 ZIP（一篇一個 TXT）；預設為課程的 rag-data.zip")
    ap.add_argument("--out", default=str(OUT_DIR / "kb"), help="輸出資料夾（會另外打包成 vector_embedding.zip）")
    ap.add_argument("--revision", default=None, help="模型 commit；預設查詢 Hugging Face 目前版本")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--question", default="我在飲料店打工，工讀生也有特休嗎？", help="建庫後試查的問題")
    args = ap.parse_args()

    course = course_assets.load(Path(args.course_dir))
    extract_rag_text, extract_metadata = course.extract_rag_text, course.extract_metadata
    archive_path = Path(args.data) if args.data else Path(args.course_dir) / "rag-data.zip"
    out_dir = Path(args.out)
    device, dtype = pick_device(args.device)
    package_versions = {n: package_metadata.version(n) for n in PACKAGE_NAMES}
    print(f"裝置：{device}／模型精度：{dtype}")

    # cells 5-9: read and normalise documents
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="rag_documents_") as tmp:
        data_dir = Path(tmp)
        txt_paths = unzip_txt(archive_path, data_dir)
        print(f"教材：{archive_path.name}／TXT 數量：{len(txt_paths)}")
        documents = []
        for number, path in enumerate(txt_paths, start=1):
            filename = path.relative_to(data_dir).as_posix()
            try:
                raw_text = path.read_text(encoding="utf-8-sig")
                if "�" in raw_text:
                    raise ValueError("文字含有替代字元，請檢查原始編碼。")
                rag_text = extract_rag_text(raw_text)
                metadata = extract_metadata(raw_text, filename)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"處理失敗：{filename}；原因：{exc}") from exc
            documents.append({"doc_id": f"doc_{number:03d}", "text": rag_text, "metadata": metadata})
    index_doc_ids = [d["doc_id"] for d in documents]
    if len(set(index_doc_ids)) != len(index_doc_ids):
        raise ValueError("doc_id 重複，請檢查編號規則。")
    for d in documents:
        missing = [k for k in ("source_url", "date") if not d["metadata"].get(k)]
        print(f"  {d['doc_id']} {d['metadata']['title']}（{len(d['text'])} 字）"
              + (f"  ⚠ 缺少 {', '.join(missing)}" if missing else ""))

    # cell 10: model (pinned to a commit for reproducibility)
    revision = args.revision or HfApi().model_info(MODEL_ID, token=False).sha
    model = load_model(revision, device, dtype)
    print(f"模型：{MODEL_ID}@{revision}／維度：{model.get_sentence_embedding_dimension()}")

    # cell 11: refuse silent truncation
    document_inputs = [DOCUMENT_TEMPLATE.format(text=d["text"]) for d in documents]
    token_limit = min(2048, int(model.max_seq_length))
    long_docs = [i for i, t in zip(index_doc_ids, document_inputs) if count_input_tokens(model, t) > token_limit]
    if long_docs:
        raise ValueError(f"以下文件超過 {token_limit} tokens，請先調整資料：{long_docs}")

    # cell 12: embeddings
    embeddings = model.encode(
        document_inputs, prompt="", batch_size=args.batch_size,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise ValueError("向量形狀與文件數不符。")
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) <= 0):
        raise ValueError("向量含有 NaN、無限值或零向量。")
    faiss.normalize_L2(embeddings)
    dimension = int(embeddings.shape[1])

    # cell 13: exact-search index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # cell 16: export the 4 files
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    for name, data in [("documents.json", documents), ("index_doc_ids.json", index_doc_ids)]:
        (out_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "format_version": "1.0",
        "created_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "model_revision": revision,
        "model_config_kwargs": MODEL_CONFIG_KWARGS,
        "dimension": dimension,
        "normalize_embeddings": True,
        "index_type": "IndexFlatIP",
        "metric": "cosine_similarity_via_normalized_inner_product",
        "document_template": DOCUMENT_TEMPLATE,
        "query_template": QUERY_TEMPLATE,
        "encode_prompt": "",
        "max_seq_length": token_limit,
        "document_count": len(documents),
        "document_unit": "one_document_one_vector",
        "document_text_includes_title": True,
        "package_versions": package_versions,
        "build_device": device,
        "build_dtype": str(dtype),
        "source_archive": archive_path.name,
        "source_archive_sha256": archive_sha256,
        "file_sha256": {
            n: hashlib.sha256((out_dir / n).read_bytes()).hexdigest()
            for n in ["index.faiss", "documents.json", "index_doc_ids.json"]
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # cell 17: reload and verify (same query vector before/after save; FAISS == numpy ranking)
    loaded = faiss.read_index(str(out_dir / "index.faiss"))
    assert json.loads((out_dir / "documents.json").read_text(encoding="utf-8")) == documents
    q = embed_question(model, args.question, token_limit, dimension)
    k = min(3, index.ntotal)
    s_before, p_before = index.search(q, k)
    s_after, p_after = loaded.search(q, k)
    np.testing.assert_array_equal(p_before, p_after)
    np.testing.assert_allclose(s_before, s_after, atol=1e-6, rtol=1e-6)
    all_scores = (embeddings @ q[0]).ravel()
    np.testing.assert_allclose(s_after[0], np.sort(all_scores)[::-1][:k], atol=1e-5, rtol=1e-5)
    print("通過：檔案完整性、文件對照、保存前後一致、精確搜尋檢查。")

    # cell 15: show a sample query
    print(f"\n試查：{args.question}")
    for rank, (pos, score) in enumerate(zip(p_after[0], s_after[0]), start=1):
        d = documents[int(pos)]
        print(f"  {rank}. [{d['doc_id']}] {d['metadata']['title']}  cos={score:.4f}")

    # cell 18: zip for Program 2
    zip_path = out_dir / "vector_embedding.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in KB_FILES:
            archive.write(out_dir / name, arcname=name)
    print(f"\n已輸出：{out_dir}/（含 {zip_path.name}）")


if __name__ == "__main__":
    main()
