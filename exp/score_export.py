"""score-export-v1 validator + normalization (harness read side, Jimmy #4 option b).

Jimmy runs a scorer with his own key, exports per-response scores as JSON following
`score-export-v1.schema.md`; the harness loads it HERE. Fail-loud: any missing/invalid
required field raises ScoreExportError — the harness never guesses direction or swallows
errors (that is how the retired-model 404 produced "all-0, looked like it ran").

Stdlib only. No network.
"""
from __future__ import annotations
import json
from typing import Optional

SCHEMA_VERSION = "score-export-v1"

# Always required (schema §2).
_REQUIRED = [
    "schema_version", "run_id", "technique", "model", "scorer_version",
    "dataset_version", "score_direction", "score_range", "rerun",
    "created_at", "scores",
]
# Conditionally required when technique == "selfcheck" (sampling consistency).
_REQUIRED_SELFCHECK = ["n_samples", "generator_model", "samples_reset"]

_DIRECTIONS = {"high_means_faithful", "high_means_hallucinated"}


class ScoreExportError(ValueError):
    """Raised when an export violates score-export-v1. Fail-loud, never guess."""


def validate(obj: dict) -> dict:
    """Validate a parsed export dict against score-export-v1. Returns it, or raises."""
    if not isinstance(obj, dict):
        raise ScoreExportError(f"export must be a JSON object, got {type(obj).__name__}")

    # Rule: schema_version must match (schema §3.1 / acceptance: schema version mismatch -> fail-loud)
    sv = obj.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ScoreExportError(f"schema_version must be {SCHEMA_VERSION!r}, got {sv!r}")

    missing = [k for k in _REQUIRED if k not in obj]
    if missing:
        raise ScoreExportError(f"missing required fields: {missing}")

    # score_direction is the most critical field (schema §3.1): required + enumerated.
    direction = obj["score_direction"]
    if direction not in _DIRECTIONS:
        raise ScoreExportError(
            f"score_direction must be one of {sorted(_DIRECTIONS)}, got {direction!r}")

    # score_range = [min, max], numeric, min < max.
    rng = obj["score_range"]
    if (not isinstance(rng, (list, tuple)) or len(rng) != 2
            or not all(isinstance(x, (int, float)) for x in rng) or rng[0] >= rng[1]):
        raise ScoreExportError(f"score_range must be [min,max] with min<max, got {rng!r}")

    if not isinstance(obj["rerun"], bool):
        raise ScoreExportError(f"rerun must be bool, got {obj['rerun']!r}")

    # selfcheck-conditional fields.
    if obj["technique"] == "selfcheck":
        miss_sc = [k for k in _REQUIRED_SELFCHECK if k not in obj]
        if miss_sc:
            raise ScoreExportError(f"technique=selfcheck requires fields: {miss_sc}")
        if not isinstance(obj["samples_reset"], bool):
            raise ScoreExportError(f"samples_reset must be bool, got {obj['samples_reset']!r}")

    # scores[] shape + range (schema §2 / §3.3 error-not-swallowed).
    scores = obj["scores"]
    if not isinstance(scores, list) or not scores:
        raise ScoreExportError("scores must be a non-empty list")
    lo, hi = rng
    for i, s in enumerate(scores):
        for k in ("bundle_id", "claim_id", "score", "error"):
            if k not in s:
                raise ScoreExportError(f"scores[{i}] missing {k!r}")
        val, err = s["score"], s["error"]
        if val is None:
            # null score MUST carry an error string (rule §3.3: errors not swallowed).
            if not isinstance(err, str) or not err:
                raise ScoreExportError(
                    f"scores[{i}] has score=null but no error string (errors must not be swallowed)")
        else:
            if not isinstance(val, (int, float)):
                raise ScoreExportError(f"scores[{i}].score must be float or null, got {val!r}")
            if not (lo <= val <= hi):
                raise ScoreExportError(f"scores[{i}].score {val} out of score_range {rng}")
    return obj


def build_export(*, technique: str, model: str, dataset_version: str, scores: list,
                 run_id: str, rerun: bool = False, scorer_version: str = "unknown",
                 score_direction: str = "high_means_faithful", score_range=(0.0, 1.0),
                 created_at: Optional[str] = None, dataset_sha: Optional[str] = None,
                 samples_reset: Optional[bool] = None, n_samples: Optional[int] = None,
                 generator_model: Optional[str] = None, temperature: Optional[float] = None) -> dict:
    """Construct + validate a score-export-v1 dict. `scores` is claim-level
    [{bundle_id, claim_id, score, error}] (keep claim granularity — do NOT pre-aggregate;
    response-level max is computed downstream so aggregation can be audited). Raises if invalid."""
    import datetime
    obj = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "technique": technique,
        "model": model, "scorer_version": scorer_version, "dataset_version": dataset_version,
        "score_direction": score_direction, "score_range": list(score_range), "rerun": rerun,
        "created_at": created_at or datetime.datetime.now().isoformat(), "scores": scores,
    }
    if dataset_sha is not None:
        obj["dataset_sha"] = dataset_sha
    if temperature is not None:
        obj["temperature"] = temperature
    if technique == "selfcheck":
        # samples_reset MUST be the real outcome (True only if reset actually succeeded).
        obj["n_samples"] = n_samples
        obj["generator_model"] = generator_model
        obj["samples_reset"] = samples_reset
    validate(obj)
    return obj


def load_export(path: str, expected_dataset_version: Optional[str] = None,
                expected_dataset_sha: Optional[str] = None) -> dict:
    """Load + validate an export file. Optionally assert it matches the harness's dataset
    version/sha (schema §3.5: scores must not be computed on a different data split)."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    validate(obj)
    if expected_dataset_version is not None and obj["dataset_version"] != expected_dataset_version:
        raise ScoreExportError(
            f"dataset_version mismatch: export={obj['dataset_version']!r} "
            f"expected={expected_dataset_version!r}")
    if expected_dataset_sha is not None and obj.get("dataset_sha") != expected_dataset_sha:
        raise ScoreExportError(
            f"dataset_sha mismatch: export={obj.get('dataset_sha')!r} expected={expected_dataset_sha!r}")
    return obj


def to_hallucination_prob(obj: dict) -> list[dict]:
    """Normalize every score to P(hallucinated) in [0,1] (schema §4). null stays null.
    Returns [{bundle_id, claim_id, p_hallu|None, error}]."""
    lo, hi = obj["score_range"]
    span = hi - lo
    flip = obj["score_direction"] == "high_means_faithful"
    out = []
    for s in obj["scores"]:
        v = s["score"]
        if v is None:
            p = None
        else:
            p = (hi - v) / span if flip else (v - lo) / span
        out.append({"bundle_id": s["bundle_id"], "claim_id": s["claim_id"],
                    "p_hallu": p, "error": s["error"]})
    return out


def coverage(obj: dict) -> float:
    """Fraction of claims with a non-null score (schema §3.3: null counts against coverage,
    NOT as 0). 0.0 if empty."""
    scores = obj["scores"]
    if not scores:
        return 0.0
    return sum(1 for s in scores if s["score"] is not None) / len(scores)


def determinism_trustworthy(obj: dict) -> bool:
    """Det-var is only meaningful on rerun exports where (for selfcheck) samples were reset
    (schema §3.4). A rerun selfcheck with samples_reset=False is NOT trustworthy."""
    if not obj["rerun"]:
        return False
    if obj["technique"] == "selfcheck" and not obj.get("samples_reset", False):
        return False
    return True
