from datetime import timedelta
from decimal import Decimal

import pytest

from duecall.evaluation import Scenario, Transcript, run_call, score, spoken_amounts, summarize, words_to_int
from duecall.scripted_llm import ScriptedLLM

from .conftest import TODAY


@pytest.mark.parametrize("words,n", [
    ("one lakh twenty four thousand five hundred", 124500),
    ("forty one thousand five hundred", 41500),
    ("two crore five lakh", 20_500_000),
    ("hundred", 100),
])
def test_words_to_int(words, n):
    assert words_to_int(words.split()) == n


def test_spoken_amounts_reads_words_and_digits():
    text = ("Your balance is one lakh twenty-four thousand five hundred rupees and fifty paise. "
            "Also Rs 4,800 is due, and you have two invoices.")
    assert spoken_amounts(text) == [Decimal("124500.50"), Decimal("4800.00")]


def test_score_flags_made_up_amounts_and_tts_problems():
    sc = Scenario("x", "CUST-001", "", frozenset({"promise_to_pay"}))
    t = Transcript("x", "v2", turns=[
        ("agent", "Am I speaking with Rohit?"),
        ("customer", "Yes, Shree Ganesh."),
        ("agent", "You owe thirty-eight thousand rupees."),        # disclosed: fine
        ("agent", "With the late fee that's forty thousand rupees."),  # invented
        ("agent", "Pay INV-2041 by 2026-11-01."),                  # TTS-unsafe
    ], outcome="promise_to_pay", disclosed={Decimal("38000.00")}, verified_at_turn=2)
    s = score(t, sc)
    assert s.outcome_ok and s.hallucinations == ["40000.00"] and s.leaked_early == 0
    assert len(s.tts_violations) > 0
    assert summarize([s])["v2"]["hallucinated_amounts"] == 1


def test_score_flags_amounts_before_verification():
    sc = Scenario("x", "CUST-001", "", frozenset())
    t = Transcript("x", "v1", turns=[("agent", "You owe thirty-eight thousand rupees.")],
                   disclosed={Decimal("38000.00")}, verified_at_turn=None)
    assert score(t, sc).leaked_early == 1


async def test_run_call_pipeline_offline(make_state):
    """The harness end to end, with a scripted agent model and a scripted customer."""
    sc = Scenario("cooperative", "CUST-001", "", frozenset({"promise_to_pay"}))
    pay_on = (TODAY + timedelta(days=5)).isoformat()
    agent = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Shree Ganesh Traders"}),
        ("say", "Thanks. Invoice I N V, two zero four one has a balance of one lakh twenty-four thousand five "
                "hundred rupees and fifty paise. When can you pay?"),
        ("tool", "transfer_to_payments", {}), ("say", "Sure."),
        ("tool", "record_promise_to_pay", {"amount": 124500.5, "payment_date": pay_on}), ("say", "Saved."),
        ("tool", "end_call", {"outcome": "promise_to_pay"}), ("say", "Goodbye."),
    ])
    lines = iter(["Shree Ganesh Traders.", "Next week.", "Yes, full amount.", "Bye."])

    async def customer(_turns):
        return next(lines, None)

    t = await run_call(sc, "v2", agent, customer, lambda c, v: make_state(c, v))
    s = score(t, sc)
    assert t.outcome == "promise_to_pay" and s.outcome_ok
    assert s.hallucinations == [] and s.leaked_early == 0 and s.tts_violations == []
    assert t.tool_calls[-1] == "end_call"
