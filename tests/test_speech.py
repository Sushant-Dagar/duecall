from datetime import date

import pytest

from duecall.speech import (SentenceBuffer, normalize_for_tts, number_to_words, speak_amount, speak_date,
                            speak_reference, unspeakable_tokens)


@pytest.mark.parametrize("n,indian,expected", [
    (0, True, "zero"),
    (15, True, "fifteen"),
    (105, True, "one hundred five"),
    (124500, True, "one lakh twenty-four thousand five hundred"),
    (12_500_000, True, "one crore twenty-five lakh"),
    (1_250_000, False, "one million two hundred fifty thousand"),
])
def test_number_to_words(n, indian, expected):
    assert number_to_words(n, indian=indian) == expected


def test_amounts_use_lakh_for_rupees_and_include_paise():
    assert speak_amount("124500.50", "INR") == \
        "one lakh twenty-four thousand five hundred rupees and fifty paise"
    assert speak_amount(4800, "SGD") == "four thousand eight hundred Singapore dollars"


def test_dates_skip_the_year_when_it_is_this_year():
    assert speak_date(date(2026, 11, 5), today=date(2026, 10, 20)) == "the fifth of November"
    assert speak_date(date(2027, 1, 22), today=date(2026, 10, 20)) == \
        "the twenty-second of January, twenty twenty-seven"
    assert speak_date(date(2026, 3, 30), today=date(2026, 1, 1)) == "the thirtieth of March"


def test_references_are_spelled_out():
    assert speak_reference("INV-2041") == "I N V, two zero four one"


def test_normalizer_cleans_llm_text():
    out = normalize_for_tts("**Please** pay Rs 4,800 for INV-3102 by 2026-11-05, 100% due.",
                            today=date(2026, 10, 20))
    assert out == ("Please pay four thousand eight hundred rupees for I N V, three one zero two by "
                   "the fifth of November, one hundred percent due.")
    assert unspeakable_tokens(out) == []


def test_sentence_buffer_never_splits_an_amount():
    buf = SentenceBuffer(today=date(2026, 10, 20))
    out = []
    for frag in ["You owe ₹1,2", "4,500. Can", " you pay", " today?"]:
        out += buf.push(frag)
    out += buf.flush()
    assert out == ["You owe one lakh twenty-four thousand five hundred rupees. ", "Can you pay today?"]
