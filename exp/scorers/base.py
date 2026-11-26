"""Scorer interface (Jimmy #6 resolution).

Decision (Emma 2026-11-27): keep the AlignedClaim contract UNCHANGED; the question is
supplied to the scorer out-of-band via `set_question(bundle["question"])` before scoring,
rather than adding a `question` field to AlignedClaim.

`reset_samples()` matters for sampling scorers (SelfCheckGPT): it must GENUINELY clear any
cached stochastic samples so a rerun draws fresh ones. Otherwise a rerun reuses old samples
and determinism-variance looks like 0 (fake). It is not "just a method name" — see
test_scorer_interface.py which proves reuse-vs-reset actually changes the scores.
"""
from __future__ import annotations
from typing import Optional


class QuestionAwareScorer:
    """Base: all scorers accept a per-bundle question out-of-band (AlignedClaim unchanged)."""

    def __init__(self) -> None:
        self._question: Optional[str] = None
        self._samples_reset: bool = True   # True until a score() consumes the reset state

    def set_question(self, question: Optional[str]) -> None:
        """Set the question context for subsequent score() calls. Call once per bundle."""
        self._question = question

    @property
    def question(self) -> Optional[str]:
        return self._question

    def reset_samples(self) -> None:
        """Clear cached stochastic samples so the next run re-samples independently.
        No-op for deterministic scorers (embedding / NLI) — they override nothing."""
        self._samples_reset = True

    @property
    def samples_reset(self) -> bool:
        """Whether samples were reset and not yet consumed by a scoring pass.
        The harness records this into the score-export `samples_reset` field."""
        return self._samples_reset

    def score(self, rec: dict) -> float:
        raise NotImplementedError
