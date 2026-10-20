import os
from datetime import timedelta
from decimal import Decimal

os.environ["DUECALL_DASHBOARD_AUTOLOAD"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from duecall.dashboard import create_app  # noqa: E402
from duecall.dialer import overdue_customers, plan_calls  # noqa: E402

from .conftest import TODAY  # noqa: E402


def test_dashboard_reports_outcomes_promises_and_latency(ledger):
    call = ledger.start_call("CUST-001", "v2", "test")
    inv = ledger.invoice("INV-2041", "CUST-001")
    ledger.record_promise(call, inv, Decimal("50000"), TODAY + timedelta(days=3), TODAY)
    ledger.end_call(call, "promise_to_pay")
    ledger.record_metric(call, "llm_ttft_ms", 420.0)
    client = TestClient(create_app(ledger))
    data = client.get("/api/summary").json()
    assert data["promised_total"] == {"n": 1, "amount": 50000.0}
    assert data["by_prompt"] == [{"prompt_version": "v2", "calls": 1, "committed": 1}]
    page = client.get("/").text
    assert "1 promises" in page and "llm_ttft_ms" in page


def test_dialer_targets_only_overdue_and_alternates_prompt_versions(ledger):
    targets = overdue_customers(ledger, TODAY)
    assert [c for c, _ in targets] == ["CUST-001", "CUST-002", "CUST-003"]
    plan = plan_calls(targets, ["v1", "v2"])
    assert [c["prompt_version"] for c in plan] == ["v1", "v2", "v1"]
    assert plan[0] == {"customer_id": "CUST-001", "phone": "+919800000001", "prompt_version": "v1"}
