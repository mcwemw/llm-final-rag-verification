"""Course materials for Program 1 / 2: download them, verify them, load the course's own code.

The course notebooks and corpus (yenlung/Generative-AI) carry no license, so this repo does NOT
redistribute them. Each user downloads them from the author's repository at a pinned commit
(`python -m rag_base.course_assets`), the files are checked against sha256 hashes, and the few
course functions our local port uses unchanged (system prompt, text extraction, the course's
citation check) are read from the downloaded notebooks at run time instead of being copied here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REPO = "yenlung/Generative-AI"
COMMIT = "655a7861f14cc1da091f8324341211e631c23a0a"  # 2026-10-27
DEFAULT_DIR = Path(__file__).resolve().parent / "course"  # gitignored

FILES = {
    "RAG_01_打造你的_RAG_系統.ipynb": "79522d0f2c01198ddc1bacf00ed7d36fd53a6f72578fa80e4131db08b5ffec5d",
    "RAG_02_使用篇.ipynb": "a26f1c4ffad3971133e536eac6c891785be95eaefae1e80678eab3435dfe2cca",
    "rag-data.zip": "f478757765fc913ce175179b4f7b107630b10517c3cd9ff4cd6675f273c98b42",
    "vector_embedding.zip": "33c283e8153e6245e915ad28391dcecad8ecd86aa61040fb39c222ac40d76cb7",
}
PROGRAM1_NB = "RAG_01_打造你的_RAG_系統.ipynb"
PROGRAM2_NB = "RAG_02_使用篇.ipynb"

# Top-level definitions taken unchanged from each notebook.
USED = {
    PROGRAM1_NB: ["extract_rag_text", "extract_metadata"],
    PROGRAM2_NB: ["SYSTEM_PROMPT", "build_context", "build_messages", "safe_api_error", "citation_note"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(course_dir: Path, names=None) -> None:
    for name in names or FILES:
        path = course_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run `python -m rag_base.course_assets` "
                                    "to download the course materials.")
        if sha256(path) != FILES[name]:
            raise ValueError(f"{path} does not match the pinned version ({REPO}@{COMMIT[:7]}).")


def download(course_dir: Path = DEFAULT_DIR) -> None:
    course_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = course_dir / name
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{urllib.parse.quote(name)}"
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"downloaded {name} does not match the pinned sha256")
        path.write_bytes(data)
        print(f"fetched {name}")
    print(f"course materials in {course_dir} ({REPO}@{COMMIT[:7]})")


def _definitions(notebook: Path, names: list[str]) -> list[ast.stmt]:
    """Top-level `def NAME` / `NAME = ...` statements from a notebook's code cells."""
    found: dict[str, ast.stmt] = {}
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                found[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        found[target.id] = node
    missing = [n for n in names if n not in found]
    if missing:
        raise ValueError(f"{notebook.name}: definitions not found: {missing}")
    return [found[n] for n in names]


def load(course_dir: Path = DEFAULT_DIR, check_hash: bool = True) -> SimpleNamespace:
    """Execute only the USED definitions of each notebook; returns them as attributes."""
    course_dir = Path(course_dir)
    if check_hash:
        verify(course_dir, list(USED))
    ns: dict = {"re": re, "json": json, "Path": Path}
    for nb, names in USED.items():
        module = ast.Module(body=_definitions(course_dir / nb, names), type_ignores=[])
        exec(compile(module, str(course_dir / nb), "exec"), ns)  # noqa: S102 — pinned, hash-checked
    return SimpleNamespace(**{n: ns[n] for names in USED.values() for n in names},
                           course_dir=course_dir, commit=COMMIT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="where to put the course materials")
    args = ap.parse_args()
    download(Path(args.dir))
    a = load(Path(args.dir))
    print("loaded from notebooks:", ", ".join(n for names in USED.values() for n in names))
    print("SYSTEM_PROMPT starts with:", a.SYSTEM_PROMPT.splitlines()[0][:30], "...")


if __name__ == "__main__":
    sys.exit(main())
