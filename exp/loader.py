"""RAGTruth loader. Public hallucination corpus (ParticleMedia/RAGTruth).

Response-level faithfulness: y_true = 1 if the response carries >=1 human
hallucination span, else 0. Source text = the grounding context:
  QA        -> source_info['passages'] (retrieved docs — the true RAG setting)
  Summary   -> source_info (string)
  Data2txt  -> source_info (dict; json-stringified)
"""
from __future__ import annotations
import json
import os

DATA = os.path.join(os.path.dirname(__file__), "data")


def _source_text(task_type: str, source_info) -> str:
    if task_type == "QA" and isinstance(source_info, dict):
        return str(source_info.get("passages", ""))
    if isinstance(source_info, dict):
        return json.dumps(source_info, ensure_ascii=False)
    return str(source_info)


def load(task_type: str = "QA", split: str = "test") -> list[dict]:
    src = {}
    with open(os.path.join(DATA, "source_info.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            src[d["source_id"]] = d
    out = []
    with open(os.path.join(DATA, "response.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            s = src.get(r["source_id"])
            if s is None:
                continue
            if task_type and s["task_type"] != task_type:
                continue
            if split and r["split"] != split:
                continue
            out.append({
                "id": r["id"],
                "source_id": r["source_id"],
                "task_type": s["task_type"],
                "source_text": _source_text(s["task_type"], s["source_info"]),
                "response": r["response"],
                "y_true": 1 if r["labels"] else 0,
                "n_spans": len(r["labels"]),
            })
    return out


if __name__ == "__main__":
    import sys
    tt = sys.argv[1] if len(sys.argv) > 1 else "QA"
    rows = load(tt, "test")
    pos = sum(r["y_true"] for r in rows)
    print(f"{tt} test: {len(rows)} responses, {pos} hallucinated "
          f"({pos/len(rows)*100:.1f}%), {len(rows)-pos} faithful")
    print("sample source len chars:", len(rows[0]["source_text"]),
          "| response len:", len(rows[0]["response"]))
