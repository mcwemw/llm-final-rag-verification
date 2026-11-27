"""Versioned gold mapping (claim -> PASS / UNSOURCED / FAIL) from RAGTruth human span labels.

Rules v1.0 is tools/build_calib_subset.py:gold_for, used by the frozen calib_v1. That file is not
modified here, so calib_v1 stays reproducible against the code hashes in its manifest.

Rules v1.1 (approved by the team lead 2026-11-27) changes exactly one thing:
    a span counts as overlapping a claim only if the overlapping characters contain at least one
    letter or digit. Overlap made only of punctuation or whitespace is not content overlap.
Motivation: some human spans start at the previous sentence's final period
(e.g. ". A higher carbon footprint ..."), which made the claim before it a false positive.

Segmentation is unchanged: tools/segment.py rules v1.0. Only the gold mapping is versioned.
"""
from build_calib_subset import BASELESS, CONFLICT, dedupe_labels, gold_for as gold_for_v10
from segment import segment

GOLD_RULES_VERSION = "1.1"


def gold_for_v11(claim, labels, response):
    s, e = claim["char_start"], claim["char_end"]
    hit, covered = [], set()
    for l in labels:
        lo, hi = max(s, l["start"]), min(e, l["end"])
        if lo < hi and any(response[i].isalnum() for i in range(lo, hi)):
            hit.append(l)
            covered.update(range(lo, hi))
    types = sorted({l["label_type"] for l in hit})
    if any(t in CONFLICT for t in types):
        verdict = "FAIL"
    elif types:
        unknown = set(types) - BASELESS
        assert not unknown, f"unknown label_type {unknown}"
        verdict = "UNSOURCED"
    else:
        verdict = "PASS"
    return {"gold_verdict": verdict, "y_true": int(bool(types)), "label_types": types,
            "overlap_ratio": round(len(covered) / (e - s), 4)}


def claims_for(resp, rules="1.1"):
    """Same fields and claim_ids as build_calib_subset.claims_for; gold from the chosen rules."""
    labels = dedupe_labels(resp["labels"])
    out = []
    for k, p in enumerate(segment(resp["response"]), 1):
        c = {"claim_id": f"ragtruth-{resp['id']}-c{k}", "response_id": resp["id"], "source_id": resp["source_id"],
             "model": resp["model"], "claim_text": p.text, "char_start": p.char_start, "char_end": p.char_end,
             "excluded": p.excluded, "exclude_reason": p.exclude_reason}
        if rules == "1.0":
            c.update(gold_for_v10(c, labels))
        elif rules == "1.1":
            c.update(gold_for_v11(c, labels, resp["response"]))
        else:
            raise ValueError(f"unknown gold rules {rules!r}")
        out.append(c)
    return out
