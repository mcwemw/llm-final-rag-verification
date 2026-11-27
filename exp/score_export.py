"""score-export-v1 ingest for the harness.

Jimmy (or any teammate) runs a scorer (RAGAS / SelfCheckGPT / NLI / …) with
their own API key, exports one JSON per (technique × model × run), and the
harness reads it here to compute metrics.

Contract: score-export-v1.schema.md (repo root).

This module is deliberately FAIL-LOUD: a malformed or under-specified export
is REJECTED with a clear error, never silently coerced. We do not guess the
score direction, we do not treat a missing field as a default, and we do not
count an errored claim as score 0.

Pure stdlib; no network, no heavy deps. Reused by exp/run_harness.py and tests.
"""
from __future__ import annotations

import json
import re
from typing import Optional

SCHEMA_VERSION = "score-export-v1"

# Top-level fields every export must carry (schema §2, "必填 ✅").
REQUIRED_TOP = [
    "schema_version", "run_id", "technique", "model", "scorer_version",
    "dataset_version", "score_direction", "score_range", "rerun",
    "created_at", "scores",
]
# Extra fields required only for sampling-consistency techniques (selfcheck).
REQUIRED_SELFCHECK = ["n_samples", "generator_model", "samples_reset"]

VALID_DIRECTIONS = {"high_means_faithful", "high_means_hallucinated"}

# A "pure response id" per Emma's Gate-2 rule: the bare RAGTruth response id
# (e.g. "12003"), NOT a decorated label like "bundle_03" or "ragtruth-12003".
# RAGTruth response ids are numeric strings; dataset identity lives in
# dataset_version + dataset_sha, not in the bundle_id.
_PURE_RESPONSE_ID = re.compile(r"^\d+$")


class ScoreExportError(ValueError):
    """Raised when an export violates score-export-v1 (fail-loud)."""


def load_export(path_or_obj, expected_dataset_version: Optional[str] = None,
                expected_dataset_sha: Optional[str] = None) -> dict:
    """Load + validate an export HEADER. Raises ScoreExportError on any
    missing/invalid required field. Returns the parsed object.

    Accepts a path (str) or an already-parsed dict (for tests).
    expected_dataset_version / expected_dataset_sha (optional): schema §3 rule 5 —
    reject an export computed on a different data version (ported from the
    release/harness loader so the canonical ingest keeps that guard)."""
    if isinstance(path_or_obj, dict):
        obj = path_or_obj
    else:
        with open(path_or_obj) as f:
            obj = json.load(f)

    if obj.get("schema_version") != SCHEMA_VERSION:
        raise ScoreExportError(
            f"schema_version must be {SCHEMA_VERSION!r}, got "
            f"{obj.get('schema_version')!r}")

    missing = [k for k in REQUIRED_TOP if k not in obj]
    if missing:
        raise ScoreExportError(f"missing required field(s): {missing}")

    if obj["score_direction"] not in VALID_DIRECTIONS:
        raise ScoreExportError(
            f"score_direction must be one of {sorted(VALID_DIRECTIONS)}, "
            f"got {obj['score_direction']!r}")

    rng = obj["score_range"]
    if (not isinstance(rng, (list, tuple)) or len(rng) != 2
            or not all(isinstance(x, (int, float)) for x in rng)
            or rng[0] >= rng[1]):
        raise ScoreExportError(
            f"score_range must be [min, max] with min < max, got {rng!r}")

    if obj["technique"] == "selfcheck":
        miss_sc = [k for k in REQUIRED_SELFCHECK if k not in obj]
        if miss_sc:
            raise ScoreExportError(
                f"selfcheck export missing required field(s): {miss_sc}")

    if not isinstance(obj["scores"], list) or not obj["scores"]:
        raise ScoreExportError("scores[] must be a non-empty list")

    if expected_dataset_version is not None and obj["dataset_version"] != expected_dataset_version:
        raise ScoreExportError(
            f"dataset_version mismatch: export={obj['dataset_version']!r} "
            f"expected={expected_dataset_version!r}")
    if expected_dataset_sha is not None and obj.get("dataset_sha") != expected_dataset_sha:
        raise ScoreExportError(
            f"dataset_sha mismatch: export={obj.get('dataset_sha')!r} "
            f"expected={expected_dataset_sha!r}")

    return obj


def _check_bundle_id(bundle_id, id_universe: Optional[set]) -> None:
    if not isinstance(bundle_id, (str, int)):
        raise ScoreExportError(f"bundle_id must be str/int, got {bundle_id!r}")
    bid = str(bundle_id)
    if not _PURE_RESPONSE_ID.match(bid):
        raise ScoreExportError(
            f"bundle_id must be a pure response id (digits only, e.g. '12003'); "
            f"got {bid!r}. Decorated ids like 'bundle_03'/'ragtruth-12003' are "
            f"rejected — this is the matching-bug guard.")
    if id_universe is not None and bid not in id_universe:
        raise ScoreExportError(
            f"bundle_id {bid!r} is not in the dataset id universe "
            f"({len(id_universe)} known response ids) — export does not align "
            f"with the data the harness loaded.")


def to_hallucination_prob(export: dict, id_universe: Optional[set] = None) -> list[dict]:
    """Normalise every score to P(hallucinated) in [0, 1], flipping by
    score_direction. Validates each row (bundle_id pure response id; in
    id_universe when provided). A null score stays None (coverage gap), it is
    NOT turned into 0.

    Returns rows: {bundle_id, claim_id, p_hallu (float|None), error}."""
    obj = load_export(export)
    lo, hi = obj["score_range"]
    span = hi - lo
    flip = obj["score_direction"] == "high_means_faithful"

    out = []
    for i, s in enumerate(obj["scores"]):
        for k in ("bundle_id", "claim_id", "score", "error"):
            if k not in s:
                raise ScoreExportError(f"scores[{i}] missing {k!r}")
        _check_bundle_id(s["bundle_id"], id_universe)
        raw = s["score"]
        if raw is None:
            p = None
        else:
            if not isinstance(raw, (int, float)):
                raise ScoreExportError(
                    f"scores[{i}].score must be a number or null, got {raw!r}")
            if raw < lo or raw > hi:
                raise ScoreExportError(
                    f"scores[{i}].score {raw} outside score_range [{lo},{hi}]")
            p = (hi - raw) / span if flip else (raw - lo) / span
        out.append({
            "bundle_id": str(s["bundle_id"]),
            "claim_id": s["claim_id"],
            "p_hallu": p,
            "error": s["error"],
        })
    return out


def coverage(prob_rows: list[dict]) -> float:
    """Fraction of claims with a real score. Errored/null claims lower
    coverage; they are NOT counted as 0 (avoids 'all-zero fakes a run')."""
    if not prob_rows:
        return 0.0
    scored = sum(1 for r in prob_rows if r["p_hallu"] is not None)
    return scored / len(prob_rows)


def determinism_trusted(export: dict) -> bool:
    """Schema section 3 rule 4 (release semantics, decided 2026-11-27):
    - rerun=false: determinism was not measured by this export -> False
      (never carry over an earlier or cached determinism result);
    - selfcheck with rerun=true & samples_reset=false fakes Det-var=0 -> False;
    - otherwise True."""
    obj = load_export(export)
    if not obj.get("rerun"):
        return False
    if obj["technique"] == "selfcheck":
        return bool(obj.get("samples_reset", False))
    return True


def response_verdicts(prob_rows: list[dict], threshold: float) -> dict:
    """Aggregate per-claim P(hallucinated) to a response-level flag, matching
    the gold rule 'a response is hallucinated if ANY claim is': a response is
    flagged (1) if any of its scored claims has p_hallu >= threshold.

    Claims with p_hallu None (errored) are skipped for the OR but DO count a
    response as 'scored'. A response with only errored claims is 'uncovered'
    and is NOT emitted (cannot be judged — do not fake it as 0).

    Returns {response_id(str): 0|1} over responses that had >=1 real score."""
    if not 0.0 <= threshold <= 1.0:
        raise ScoreExportError(f"threshold must be in [0,1], got {threshold}")
    scored, flag = {}, {}
    for r in prob_rows:
        rid = str(r["bundle_id"])
        p = r["p_hallu"]
        if p is None:
            scored.setdefault(rid, False)
            continue
        scored[rid] = True
        flag[rid] = flag.get(rid, 0) or (1 if p >= threshold else 0)
    return {rid: flag.get(rid, 0) for rid, has in scored.items() if has}


def summarize(export, id_universe: Optional[set] = None, threshold: float = 0.5) -> dict:
    """Validate + summarize an export without computing metrics vs gold (which
    needs the dataset). Fail-loud on any contract violation."""
    obj = load_export(export)
    rows = to_hallucination_prob(obj, id_universe)
    verdicts = response_verdicts(rows, threshold)
    return {
        "run_id": obj["run_id"],
        "technique": obj["technique"],
        "model": obj["model"],
        "dataset_version": obj["dataset_version"],
        "n_claims": len(rows),
        "coverage": round(coverage(rows), 4),
        "determinism_trusted": determinism_trusted(obj),
        "n_responses_scored": len(verdicts),
        "n_responses_flagged": sum(verdicts.values()),
        "threshold": threshold,
    }


if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(
        description="Validate + summarize a score-export-v1 file (fail-loud).")
    ap.add_argument("export", help="path to a score-export-v1 JSON file")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(hallucinated) threshold for the response-level OR")
    ap.add_argument("--ids-from", default=None,
                    help="optional responses.jsonl to build the id universe "
                         "(enforces bundle_id ∈ dataset)")
    args = ap.parse_args()

    universe = None
    if args.ids_from:
        universe = set()
        with open(args.ids_from) as f:
            for line in f:
                line = line.strip()
                if line:
                    universe.add(str(_json.loads(line)["id"]))

    try:
        print(_json.dumps(summarize(args.export, universe, args.threshold), indent=2))
    except ScoreExportError as e:
        raise SystemExit(f"REJECTED (score-export-v1 violation): {e}")
