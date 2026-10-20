from datetime import timedelta
from decimal import Decimal

import pytest

from duecall.ledger import LedgerError
from duecall.policy import PolicyIndex

from .conftest import TODAY


@pytest.mark.parametrize("cid,said", [
    ("CUST-001", "Shri Ganesh Traders"),
    ("CUST-001", "Sri Ganesh traders"),
    ("CUST-001", "This is Rohit Mehta from Shri Ganesh Traders"),
    ("CUST-001", "yes, Rohit here, Shree Ganesh Traders accounts"),
    ("CUST-002", "nova retail private limited"),
    ("CUST-002", "I'm Anjali from Nova Retail"),
    ("CUST-003", "Daniel at Bluepeak Logistics"),
])
def test_identity_match_accepts_stt_variants_and_full_sentences(ledger, cid, said):
    assert ledger.match_company(cid, said)


@pytest.mark.parametrize("cid,said", [
    ("CUST-001", "Nova Retail"),
    ("CUST-001", "Ganesh Traders"),           # missing a word of the name
    ("CUST-001", "Rohit Mehta"),               # a person, not the company
    ("CUST-001", "Sikh and H traders"),        # mis-transcribed: ask again
    ("CUST-002", "Sunrise Hardware"),
    ("CUST-002", ""),
    ("CUST-999", "Shree Ganesh Traders"),
])
def test_identity_match_rejects_other_names(ledger, cid, said):
    assert not ledger.match_company(cid, said)


def test_invoices_are_scoped_to_the_customer(ledger):
    with pytest.raises(LedgerError):
        ledger.invoice("INV-1988", "CUST-001")  # belongs to CUST-002
    assert [i.number for i in ledger.open_invoices("CUST-001")] == ["INV-2041", "INV-2077"]
    assert ledger.invoice("inv-1988", "CUST-002").balance == Decimal("66250")


@pytest.mark.parametrize("amount,days,msg", [
    ("0", 3, "more than zero"),
    ("200000", 3, "more than the balance"),
    ("1000", -1, "already passed"),
    ("1000", 15, "within 14 days"),
])
def test_promise_rules(ledger, amount, days, msg):
    inv = ledger.invoice("INV-2041", "CUST-001")
    with pytest.raises(LedgerError, match=msg):
        ledger.record_promise("c1", inv, Decimal(amount), TODAY + timedelta(days=days), TODAY)
    assert ledger.conn.execute("SELECT COUNT(*) FROM promises").fetchone()[0] == 0


def test_plan_rules(ledger):
    small = ledger.invoice("INV-3102", "CUST-003")
    with pytest.raises(LedgerError, match="too small"):
        ledger.create_plan("c1", small, 2, TODAY, TODAY)
    big = ledger.invoice("INV-2041", "CUST-001")
    with pytest.raises(LedgerError, match="two to 3"):
        ledger.create_plan("c1", big, 6, TODAY, TODAY)
    assert ledger.create_plan("c1", big, 3, TODAY + timedelta(days=5), TODAY) == Decimal("41500.17")


@pytest.mark.parametrize("question,expected", [
    ("what is the late fee if I pay after the due date", "late-fees"),
    ("can I pay by UPI", "payment-methods"),
    ("can I split it into instalments", "payment-plans"),
    ("I already paid this last week", "already-paid"),
    ("when is your finance team available", "contact-hours"),
])
def test_policy_retrieval_top1(question, expected):
    assert PolicyIndex().search(question)[0].id == expected


def test_policy_retrieval_returns_nothing_for_off_topic():
    assert PolicyIndex().search("what is the weather in Shimla") == []
