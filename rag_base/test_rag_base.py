"""Offline tests for rag_base (no course download, no API key, no embedding model)."""
import json
from pathlib import Path

import pytest

from rag_base import course_assets
from rag_base.program2_answer import SCHEMA_PATH, cited_ids_in_order, to_answer_bundle


def _notebook(path: Path, cells: list[str]) -> Path:
    nb = {"cells": [{"cell_type": "code", "source": c.splitlines(keepends=True)} for c in cells]
           + [{"cell_type": "markdown", "source": ["# not code"]}]}
    path.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def fake_course(tmp_path):
    _notebook(tmp_path / course_assets.PROGRAM1_NB, [
        "!pip install something\nimport re\n"
        "def extract_rag_text(raw_text):\n    return raw_text.strip().upper()\n"
        "print('cell side effect must not run')\n",
        "def extract_metadata(raw_text, filename):\n    return {'filename': Path(filename).name}\n",
    ])
    _notebook(tmp_path / course_assets.PROGRAM2_NB, [
        "SYSTEM_PROMPT = 'you are a test'\n"
        "def build_context(results):\n    return json.dumps([r['doc_id'] for r in results])\n"
        "def build_messages(question, results):\n"
        "    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': build_context(results) + question}]\n"
        "QUESTION = 'not loaded'\n",
        "def safe_api_error(exc):\n    return 'hidden'\n"
        "def citation_note(answer, results):\n"
        "    return 'cited' if re.findall(r'\\[(doc_[0-9]+)\\]', answer) else 'none'\n",
    ])
    return tmp_path


class TestCourseAssets:
    def test_loads_only_the_used_definitions(self, fake_course, capsys):
        a = course_assets.load(fake_course, check_hash=False)
        assert a.extract_rag_text(" x ") == "X"
        assert a.extract_metadata("", "a/b.txt") == {"filename": "b.txt"}
        msgs = a.build_messages("?", [{"doc_id": "doc_001"}])
        assert msgs[0]["content"] == "you are a test" and msgs[1]["content"] == '["doc_001"]?'
        assert a.citation_note("see [doc_001]", []) == "cited"
        assert a.safe_api_error(Exception()) == "hidden"
        assert not hasattr(a, "QUESTION")
        assert "side effect" not in capsys.readouterr().out  # only definitions are executed

    def test_missing_definition_raises(self, tmp_path, fake_course):
        _notebook(fake_course / course_assets.PROGRAM2_NB, ["SYSTEM_PROMPT = 'x'\n"])
        with pytest.raises(ValueError, match="definitions not found"):
            course_assets.load(fake_course, check_hash=False)

    def test_hash_check(self, fake_course):
        with pytest.raises(ValueError, match="pinned version"):
            course_assets.load(fake_course)  # fake notebooks do not match the pinned sha256

    def test_missing_files_point_to_download(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="course_assets"):
            course_assets.verify(tmp_path)

    def test_pinned_commit_and_hashes_look_valid(self):
        assert len(course_assets.COMMIT) == 40
        assert all(len(h) == 64 for h in course_assets.FILES.values())
        assert set(course_assets.USED) <= set(course_assets.FILES)


DOCS = [{"doc_id": "doc_003", "text": "最低工資每小時196元。"},
        {"doc_id": "doc_010", "text": "部分工時也有特休。"},
        {"doc_id": "doc_016", "text": "工讀生依法享有勞動權益。"}]


class TestAnswerBundle:
    @pytest.mark.parametrize("answer,expected", [
        ("見 [doc_003]。", ["doc_003"]),
        ("見【doc_016】與〔doc_010〕。", ["doc_016", "doc_010"]),
        ("（doc_010）［doc_003］", ["doc_010", "doc_003"]),
        ("[doc_003, doc_010] 以及 [doc_003]", ["doc_003", "doc_010"]),
        ("沒有引用。", []),
    ])
    def test_cited_ids_any_bracket_style(self, answer, expected):
        assert cited_ids_in_order(answer) == expected

    def test_bundle_spans_cover_whole_document(self):
        b, unknown = to_answer_bundle("q?", "工讀生有權益【doc_016】，時薪196元[doc_003]。", DOCS)
        assert [c["doc_id"] for c in b["citations"]] == ["doc_016", "doc_003"]
        for c in b["citations"]:
            doc = next(d for d in DOCS if d["doc_id"] == c["doc_id"])
            assert c["span_start"] == 0 and c["span_end"] == len(doc["text"])
            assert doc["text"][c["span_start"]:c["span_end"]] == c["text"]
        assert [r["doc_id"] for r in b["retrieved"]] == ["doc_003", "doc_010", "doc_016"]
        assert unknown == [] and b["schema_version"] == "1.1"

    def test_fabricated_citation_reported_not_included(self):
        b, unknown = to_answer_bundle("q?", "見 [doc_099]。", DOCS)
        assert b["citations"] == [] and unknown == ["doc_099"]

    def test_bundle_validates_against_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        b, _ = to_answer_bundle("q?", "見【doc_016】。", DOCS)
        jsonschema.validate(b, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
