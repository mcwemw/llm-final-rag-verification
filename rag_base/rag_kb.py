"""Knowledge-base helpers for the local ports of course Program 1 / Program 2 (Block 1).

Adapted (rewritten for local use) from the course notebooks, yenlung/Generative-AI @655a786:
  RAG_01_打造你的_RAG_系統.ipynb  -> embedding / search
  RAG_02_使用篇.ipynb            -> KB loading + integrity checks
Same model, templates, checks and retrieval behaviour; Colab-specific I/O replaced by local paths.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
MODEL_CONFIG_KWARGS = {"vision_config": None, "audio_config": None}
DOCUMENT_TEMPLATE = "title: none | text: {text}"
QUERY_TEMPLATE = "task: search result | query: {question}"
KB_FILES = ["index.faiss", "documents.json", "index_doc_ids.json", "config.json"]


def pick_device(device: str | None = None) -> tuple[str, torch.dtype]:
    """Same rule as the notebooks: bf16 only on a CUDA GPU that supports it, else fp32."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    return device, dtype


def load_model(revision: str, device: str, dtype: torch.dtype) -> SentenceTransformer:
    try:
        return SentenceTransformer(
            MODEL_ID, revision=revision, token=False, device=device,
            config_kwargs=MODEL_CONFIG_KWARGS,
            model_kwargs={"dtype": dtype, "attn_implementation": "eager"},
        )
    except Exception as exc:
        raise RuntimeError(
            "EmbeddingGemma 2 載入失敗。請確認網路、套件版本；若 GPU 記憶體不足，加 --device cpu。"
            "未替換成其他模型。"
        ) from exc


class KnowledgeBase:
    """A loaded, integrity-checked KB (Program 2 cells 5-8)."""

    def __init__(self, kb_path: str | Path, device: str | None = None):
        kb_path = Path(kb_path)
        raw = self._read_files(kb_path)
        self.config = json.loads(raw["config.json"])
        self._check_config(raw)
        self.documents = json.loads(raw["documents.json"])
        self.index_doc_ids = json.loads(raw["index_doc_ids.json"])
        self._check_documents()
        self.documents_by_id = {d["doc_id"]: d for d in self.documents}

        index = faiss.deserialize_index(np.frombuffer(raw["index.faiss"], dtype=np.uint8))
        if not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("FAISS 索引類型不相容。")
        self.dimension = self.config["dimension"]
        if index.d != self.dimension or not (
            index.ntotal == len(self.documents) == len(self.index_doc_ids) == self.config["document_count"]
        ):
            raise ValueError("索引維度或文件數不一致。")
        self.index = index

        self.device, self.dtype = pick_device(device)
        self.model = load_model(self.config["model_revision"], self.device, self.dtype)
        if int(self.model.get_sentence_embedding_dimension()) != self.dimension:
            raise ValueError("載入模型的向量維度與索引不符。")
        self.token_limit = int(self.config["max_seq_length"])
        self.model.max_seq_length = self.token_limit

    # ── loading / checks ────────────────────────────────────────────────────
    @staticmethod
    def _read_files(kb_path: Path) -> dict[str, bytes]:
        if kb_path.is_dir():
            return {name: (kb_path / name).read_bytes() for name in KB_FILES}
        with zipfile.ZipFile(kb_path) as archive:
            names = archive.namelist()
            for name in KB_FILES:
                if names.count(name) != 1:
                    raise ValueError(f"缺少或重複 {name}；請用 Program 1 匯出的知識庫，而非原始教材。")
            return {name: archive.read(name) for name in KB_FILES}

    def _check_config(self, raw: dict[str, bytes]) -> None:
        c = self.config
        if c.get("format_version") != "1.0":
            raise ValueError("不支援的知識庫格式版本。")
        if c.get("model_id") != MODEL_ID:
            raise ValueError("本課程只載入 Program 1 使用的 EmbeddingGemma 2。")
        if not re.fullmatch(r"[0-9a-f]{40}", c.get("model_revision", "")):
            raise ValueError("缺少有效的模型 commit；請用 Program 1 重新匯出。")
        if c.get("model_config_kwargs") != MODEL_CONFIG_KWARGS:
            raise ValueError("本教材預期為純文字模型設定。")
        if c.get("normalize_embeddings") is not True or c.get("index_type") != "IndexFlatIP":
            raise ValueError("本教材需要正規化向量與 IndexFlatIP。")
        if c.get("metric") != "cosine_similarity_via_normalized_inner_product":
            raise ValueError("知識庫的相似度設定不相容。")
        if c.get("query_template") != QUERY_TEMPLATE or c.get("encode_prompt") != "":
            raise ValueError("查詢前綴與 Program 1 規格不同。")
        for name in ["index.faiss", "documents.json", "index_doc_ids.json"]:
            if hashlib.sha256(raw[name]).hexdigest() != c.get("file_sha256", {}).get(name):
                raise ValueError(f"檔案完整性檢查失敗：{name}；請重新匯出。")

    def _check_documents(self) -> None:
        docs, ids = self.documents, self.index_doc_ids
        if not isinstance(docs, list) or not docs or not isinstance(ids, list):
            raise ValueError("文件及編號對照必須是非空清單。")
        for doc in docs:
            if not isinstance(doc, dict) or not isinstance(doc.get("doc_id"), str):
                raise ValueError("文件缺少 doc_id。")
            if not isinstance(doc.get("text"), str) or not doc["text"].strip():
                raise ValueError(f"{doc['doc_id']} 缺少正文。")
            if not isinstance(doc.get("metadata"), dict):
                raise ValueError(f"{doc['doc_id']} 缺少 metadata。")
        if len({d["doc_id"] for d in docs}) != len(docs) or len(set(ids)) != len(ids):
            raise ValueError("文件編號重複。")
        if set(ids) != {d["doc_id"] for d in docs}:
            raise ValueError("索引編號對照與文件不一致。")

    # ── retrieval (Program 2 cell 8) ────────────────────────────────────────
    def embed_question(self, question: str) -> np.ndarray:
        return embed_question(self.model, question, self.token_limit, self.dimension)

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError("top_k 必須是大於0的整數。")
        query_vector = self.embed_question(question)
        scores, positions = self.index.search(query_vector, min(int(top_k), self.index.ntotal))
        results = []
        for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1):
            if position < 0 or position >= len(self.index_doc_ids):
                raise ValueError("索引傳回不正確的文件位置。")
            doc_id = self.index_doc_ids[int(position)]
            results.append({"rank": rank, "score": float(score), **self.documents_by_id[doc_id]})
        return results


def count_input_tokens(model: SentenceTransformer, text: str) -> int:
    return len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def embed_question(model: SentenceTransformer, question: str, token_limit: int, dimension: int) -> np.ndarray:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("請輸入非空白問題。")
    query_input = QUERY_TEMPLATE.format(question=question.strip())
    if count_input_tokens(model, query_input) > token_limit:
        raise ValueError("問題超過模型 token 上限，請縮短問題。")
    vector = model.encode(
        [query_input], prompt="", convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    if vector.shape != (1, dimension) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
        raise ValueError("問題向量維度或數值不正確。")
    faiss.normalize_L2(vector)
    return vector
