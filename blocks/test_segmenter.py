"""Tests for Block 3 — Claim Segmenter. Rule set locked 2026-11-24."""
import pytest
from blocks.segmenter import segment, Claim


# ── basic split ───────────────────────────────────────────────────────────────

def test_sentence_split_multiple():
    answer = "Precision measures exactness of relevant items. Recall measures coverage of relevant items."
    claims = segment(answer)
    texts = [c.claim_text for c in claims]
    assert len(claims) == 2
    assert any("Precision" in t for t in texts)
    assert any("Recall" in t for t in texts)


def test_single_sentence_returned():
    answer = "Qualified distributions from a Roth IRA are tax-free."
    claims = segment(answer)
    assert len(claims) == 1
    assert "Qualified distributions" in claims[0].claim_text


def test_exclamation_split():
    answer = "This claim is definitely true! That statement is also clearly valid here."
    claims = segment(answer)
    assert len(claims) == 2


# ── connective split ──────────────────────────────────────────────────────────

def test_but_connective_split():
    answer = "The Kindle supports wireless, but the Paperwhite does not support wireless."
    claims = segment(answer)
    texts = [c.claim_text for c in claims]
    assert len(claims) >= 2
    assert any("Kindle supports" in t for t in texts)


def test_whereas_connective_split():
    answer = "ATVs have an open design, whereas UTVs have a closed protective cab structure."
    claims = segment(answer)
    assert len(claims) >= 2


def test_while_connective_split():
    answer = "Pine nuts are edible, while pine resin is used industrially for applications."
    claims = segment(answer)
    assert len(claims) >= 2


def test_although_connective_split():
    answer = "Although pine nuts are costly, they offer nutritional value for consumers."
    claims = segment(answer)
    assert len(claims) >= 1


def test_and_not_split():
    answer = "Roth IRA allows tax-free growth and tax-free withdrawals in retirement."
    claims = segment(answer)
    assert len(claims) == 1, "'and' should not trigger a split"


# ── semicolon split ───────────────────────────────────────────────────────────

def test_semicolon_split():
    answer = "ATVs are for off-road use; UTVs are for utility purposes in agriculture."
    claims = segment(answer)
    assert len(claims) == 2


# ── meta-sentence filter ──────────────────────────────────────────────────────

def test_meta_based_on_filtered():
    answer = "Based on the provided passage, the tax rate is 20%. The higher band reaches 500k."
    claims = segment(answer)
    texts = [c.claim_text for c in claims]
    assert all("Based on" not in t for t in texts)
    assert any("500k" in t or "higher band" in t for t in texts)


def test_meta_according_to_filtered():
    answer = "According to the given information, prices rose 10%. Financial markets remained highly volatile."
    claims = segment(answer)
    texts = [c.claim_text for c in claims]
    assert all("According to" not in t for t in texts)
    assert any("volatile" in t for t in texts)


def test_meta_passage_states_filtered():
    answer = "The passage states that harvesting requires a permit. Permits cost 50 dollars."
    claims = segment(answer)
    texts = [c.claim_text for c in claims]
    assert all("passage states" not in t for t in texts)


# ── short piece filter ────────────────────────────────────────────────────────

def test_short_pieces_filtered():
    answer = "Yes. Roth IRA earnings can be withdrawn tax-free after five years of holding."
    claims = segment(answer)
    texts = [c.claim_text for c in claims]
    assert all(len(t.split()) >= 4 for t in texts)


# ── newline and bullet split ──────────────────────────────────────────────────

def test_newline_split():
    answer = "First point about Roth IRA rules here.\nSecond point about withdrawal limits."
    claims = segment(answer)
    assert len(claims) == 2


def test_bullet_split():
    answer = "- Roth IRA allows tax-free withdrawals after age threshold.\n- Contributions are not deductible from taxes."
    claims = segment(answer)
    texts = [c.claim_text for c in claims]
    assert all(not t.startswith("-") for t in texts)
    assert len(claims) == 2


def test_bullet_not_in_claim():
    answer = "- Roth IRA allows tax-free withdrawals after the threshold."
    claims = segment(answer)
    assert not claims[0].claim_text.startswith("-")


# ── character offset correctness ──────────────────────────────────────────────

def test_char_offsets_valid():
    answer = "Qualified distributions are tax-free. Age limits do not apply to contributions."
    claims = segment(answer)
    for c in claims:
        assert 0 <= c.char_start < c.char_end <= len(answer)


def test_char_offset_first_claim():
    answer = "Roth IRA accounts must be designated as such at setup."
    claims = segment(answer)
    assert claims[0].char_start == 0


# ── determinism ───────────────────────────────────────────────────────────────

def test_determinism_three_runs():
    answer = (
        "Precision measures exactness. "
        "Recall measures coverage of relevant results. "
        "F1 balances both metrics together."
    )
    r1 = [(c.claim_text, c.char_start) for c in segment(answer)]
    r2 = [(c.claim_text, c.char_start) for c in segment(answer)]
    r3 = [(c.claim_text, c.char_start) for c in segment(answer)]
    assert r1 == r2 == r3


# ── edge cases ────────────────────────────────────────────────────────────────

def test_empty_string():
    assert segment("") == []


def test_whitespace_only():
    assert segment("   \n\t  ") == []


def test_single_short_sentence_filtered():
    assert segment("Yes.") == []
