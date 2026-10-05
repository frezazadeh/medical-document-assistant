from app.agent.citations import check_citations, cited_refs
from app.context import NOT_FOUND_ANSWER, Source

CRITERIA = (
    "Randomization was stratified by baseline eGFR. Key inclusion criteria: age 18 to 80 "
    "years; mean seated SBP of at least 140 mmHg."
)
POPULATION = (
    "A total of 612 people were screened and 412 participants were randomized. "
    "Overall, 381 participants (92.5%) completed the 12-week treatment period."
)
SAFETY = (
    "Hyperkalemia occurred in 10 participants (9.7%) on 10 mg. Serious adverse events were "
    "reported in 9 participants. No deaths occurred during the study."
)


def source(ref: int, text: str, page: int | None = None) -> Source:
    return Source(ref, f"d1:{ref}", "d1", "report.pdf", page=page or ref, text=text, score=0.8)


SOURCES = [source(3, CRITERIA, page=2), source(4, POPULATION, page=3), source(5, SAFETY, page=5)]


def test_markers_are_parsed_in_order_without_duplicates():
    assert cited_refs("Met at week 12 [2]. Also [1, 3] and again [2].") == [2, 1, 3]


def test_correct_citation_is_left_alone():
    check = check_citations("412 participants were randomized [4].", SOURCES)

    assert check.answer == "412 participants were randomized [4]."
    assert [s.page for s in check.cited] == [3]
    assert check.repairs == []
    assert check.grounded


def test_citation_pointing_at_the_wrong_passage_is_moved_to_the_right_one():
    # Seen in the UI: the number is in source 4, the model wrote [3].
    check = check_citations("412 participants were randomized [3].", SOURCES)

    assert check.answer == "412 participants were randomized [4]."
    assert [s.page for s in check.cited] == [3]
    assert check.repairs == [
        {"sentence": "412 participants were randomized [3].", "from": [3], "to": 4}
    ]
    assert check.grounded


def test_sentence_without_numbers_is_moved_when_the_words_clearly_match_elsewhere():
    check = check_citations("No deaths occurred during the study [3].", SOURCES)

    assert check.answer == "No deaths occurred during the study [5]."
    assert check.grounded


def test_only_the_wrong_sentence_is_changed():
    answer = "412 participants were randomized [4]. Hyperkalemia occurred in 9.7% on 10 mg [3]."

    check = check_citations(answer, SOURCES)

    assert check.answer == (
        "412 participants were randomized [4]. Hyperkalemia occurred in 9.7% on 10 mg [5]."
    )
    assert [s.ref for s in check.cited] == [4, 5]


def test_number_that_is_in_no_source_makes_the_answer_ungrounded():
    check = check_citations("450 participants were randomized [4].", SOURCES)

    assert check.answer.endswith("[4].")
    assert check.unsupported == ["450 participants were randomized [4]."]
    assert not check.grounded


def test_sentence_drawing_on_two_sources_keeps_both_citations():
    answer = "412 were randomized and 10 participants had hyperkalemia (9.7%) [4][5]."

    check = check_citations(answer, SOURCES)

    assert check.answer == answer
    assert check.grounded


def test_invented_id_is_replaced_when_a_source_clearly_supports_the_sentence():
    check = check_citations("412 participants were randomized [9].", SOURCES)

    assert check.answer == "412 participants were randomized [4]."
    assert check.unknown_refs == []
    assert check.grounded


def test_invented_id_without_support_is_reported():
    check = check_citations("The drug is taken with food [9].", SOURCES)

    assert check.unknown_refs == [9]
    assert not check.grounded


def test_list_numbering_is_not_treated_as_a_fact():
    answer = "1. 412 participants were randomized [4].\n2. No deaths occurred [5]."

    check = check_citations(answer, SOURCES)

    assert check.unsupported == []
    assert check.grounded


def test_answer_without_citations_is_not_grounded():
    assert not check_citations("412 participants were randomized.", SOURCES).grounded


def test_refusal_counts_as_grounded():
    check = check_citations(NOT_FOUND_ANSWER, SOURCES)

    assert check.cited == []
    assert check.grounded


def test_digits_inside_units_and_names_are_not_numbers():
    sources = [source(1, "eGFR fell by 3 mL/min/1.73 m2 through CYP11B2 inhibition.")]

    check = check_citations("Two enzymes were tested in 2 participants [1].", sources)

    assert check.unsupported == ["Two enzymes were tested in 2 participants [1]."]
