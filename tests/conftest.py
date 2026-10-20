from datetime import date

import pytest

from duecall.agents import CallState
from duecall.ledger import Ledger

TODAY = date(2026, 10, 20)


@pytest.fixture
def ledger():
    led = Ledger(":memory:")
    led.seed_demo(TODAY)
    return led


@pytest.fixture
def make_state(ledger):
    def _make(customer_id="CUST-001", version="v2"):
        call_id = ledger.start_call(customer_id, version, "test")
        return CallState(ledger=ledger, customer_id=customer_id, call_id=call_id, today=TODAY,
                         prompt_version=version)
    return _make
