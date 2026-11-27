"""Unit tests for tools/segment.py (shared Block 3 spec).  Run: python3 tools/test_segment.py  (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment import segment  # noqa: E402


def texts(answer, include_excluded=False):
    return [p.text for p in segment(answer) if include_excluded or not p.excluded]


def test_offsets_invariant():
    a = "UTVs seat two, while ATVs seat one rider only. Prices vary a lot by region; check local dealers first."
    for p in segment(a):
        assert a[p.char_start:p.char_end] == p.text


def test_sentence_and_semicolon_and_connective():
    a = "UTVs have side-by-side seating, while ATVs have a single seat. ATVs cost less overall; UTVs carry more cargo."
    assert texts(a) == ["UTVs have side-by-side seating", "ATVs have a single seat.",
                        "ATVs cost less overall", "UTVs carry more cargo."]


def test_and_is_not_a_boundary():
    assert texts("We measured accuracy and cost, and both improved a lot.") == [
        "We measured accuracy and cost, and both improved a lot."]


def test_abbreviation_is_not_a_boundary():
    assert texts("Mix flour, e.g. rye flour, with plenty of water.") == ["Mix flour, e.g. rye flour, with plenty of water."]
    assert texts("Heat 2 tbsp. of olive oil in a large pot.") == ["Heat 2 tbsp. of olive oil in a large pot."]


def test_bullets_and_markers():
    a = "1. Preheat the oven to 150C.\n2. Bake the bread for an hour.\n- (Passage 2)"
    assert texts(a) == ["Preheat the oven to 150C.", "Bake the bread for an hour."]


def test_meta_excluded_and_attribution_stripped():
    a = ("Based on the provided passages, the answer is:\nAccording to passage 3, ATVs are harder to ride.\n"
         "I hope this helps!")
    ps = segment(a)
    assert [(p.text, p.exclude_reason) for p in ps] == [
        ("Based on the provided passages, the answer is:", "meta"),
        ("ATVs are harder to ride.", None),
        ("I hope this helps!", "meta")]


def test_passage_as_subject_is_attribution_not_marker():
    assert texts("Passage 3 suggests that you can reheat the chicken in the microwave.") == [
        "you can reheat the chicken in the microwave."]
    assert texts("Passage 2: Chicken can be reheated in the microwave.") == [
        "Chicken can be reheated in the microwave."]


def test_colon_merged_with_short_followup():
    assert texts("The butcher shop phone number is:\n(555) 123-4567") == [
        "The butcher shop phone number is:\n(555) 123-4567"]


def test_short_piece_excluded():
    ps = segment("Pine nuts grow very slowly. They need years.")
    assert [(p.text, p.exclude_reason) for p in ps] == [("Pine nuts grow very slowly.", None),
                                                       ("They need years.", "short")]


def test_deterministic():
    a = "A is true, but B is false; C holds here.\n- D follows from it.\nSure, happy to help!"
    assert all(segment(a) == segment(a) for _ in range(3))


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
