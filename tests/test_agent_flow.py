"""End-to-end conversation tests through the real LiveKit AgentSession, with a
scripted LLM in place of a model. These pin down what the code guarantees
whatever the model does: tool gating, handoffs, validation and ledger writes."""

from datetime import timedelta
from decimal import Decimal

from livekit.agents import AgentSession
from livekit.agents.voice.run_result import AgentHandoffEvent, FunctionCallOutputEvent

from duecall.agents import CollectionsAgent, DisputesAgent, PaymentsAgent
from duecall.scripted_llm import ScriptedLLM

from .conftest import TODAY


def outputs(result):
    return [e.item for e in result.events if isinstance(e, FunctionCallOutputEvent)]


async def test_promise_to_pay_flow(make_state, ledger):
    state = make_state("CUST-001")
    pay_on = (TODAY + timedelta(days=7)).isoformat()
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Shri Ganesh Traders"}),
        ("say", "Thanks. Invoice I N V, two zero four one is overdue. When can you pay?"),
        ("tool", "transfer_to_payments", {"invoice_number": "INV-2041"}),
        ("say", "Sure, let's set a date."),
        ("tool", "record_promise_to_pay", {"amount": 124500.50, "payment_date": pay_on}),
        ("say", "Saved."),
        ("tool", "end_call", {"outcome": "promise_to_pay"}),
        ("say", "Thank you, goodbye."),
    ])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))

        r1 = await session.run(user_input="This is Shri Ganesh Traders accounts.")
        out = outputs(r1)[0]
        assert not out.is_error and "one lakh twenty-four thousand five hundred rupees" in out.output
        assert "twenty-five days overdue" in out.output

        r2 = await session.run(user_input="I can pay next week.")
        assert any(isinstance(e, AgentHandoffEvent) for e in r2.events)
        assert isinstance(session.current_agent, PaymentsAgent)

        r3 = await session.run(user_input="Yes, the full amount, that's right.")
        assert not outputs(r3)[0].is_error

        await session.run(user_input="Thanks, bye.")

    row = ledger.conn.execute("SELECT * FROM promises").fetchone()
    assert (row["invoice_number"], Decimal(row["amount"]), row["promised_date"]) == ("INV-2041", Decimal("124500.50"), pay_on)
    call = ledger.conn.execute("SELECT outcome FROM calls WHERE id = ?", (state.call_id,)).fetchone()
    assert call["outcome"] == "promise_to_pay"
    assert Decimal("124500.50") in state.disclosed_amounts


async def test_no_account_data_before_verification(make_state):
    state = make_state("CUST-001")
    model = ScriptedLLM([("tool", "get_open_invoices", {}), ("say", "Which company are you with?"),
                         ("tool", "transfer_to_payments", {"invoice_number": "INV-2041"}), ("say", "Sorry.")])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        r1 = await session.run(user_input="How much do I owe?")
        r2 = await session.run(user_input="Just take my payment.")
    for r in (r1, r2):
        out = outputs(r)[0]
        assert out.is_error and "not verified" in out.output
    assert isinstance(session.current_agent, CollectionsAgent)
    assert state.disclosed_amounts == set()


async def test_failed_verification_discloses_nothing(make_state):
    state = make_state("CUST-002")
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Bluepeak Logistics"}), ("say", "Could you repeat that?"),
        ("tool", "verify_identity", {"company_name": "Ganesh Traders"}), ("say", "I'll email instead."),
    ])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        await session.run(user_input="Bluepeak.")
        r = await session.run(user_input="Ganesh Traders.")
    assert "Do not share any account details" in outputs(r)[0].output
    assert state.outcome == "verification_failed" and not state.verified
    assert state.disclosed_amounts == set()


async def test_invalid_promise_is_rejected_and_not_saved(make_state, ledger):
    state = make_state("CUST-001")
    late = (TODAY + timedelta(days=40)).isoformat()
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Shree Ganesh Traders"}), ("say", "ok"),
        ("tool", "transfer_to_payments", {}), ("say", "ok"),
        ("tool", "record_promise_to_pay", {"amount": 124500.5, "payment_date": late}),
        ("say", "I can only note a date within fourteen days. Would a payment plan help?"),
        ("tool", "record_promise_to_pay", {"amount": 999999, "payment_date": TODAY.isoformat()}),
        ("say", "That is more than the balance."),
    ])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        await session.run(user_input="Shree Ganesh Traders")
        await session.run(user_input="I want to pay")
        r1 = await session.run(user_input="End of November.")
        r2 = await session.run(user_input="Put down ten lakh then.")
    assert outputs(r1)[0].is_error and "within 14 days" in outputs(r1)[0].output
    assert outputs(r2)[0].is_error and "more than the balance" in outputs(r2)[0].output
    assert ledger.conn.execute("SELECT COUNT(*) FROM promises").fetchone()[0] == 0
    assert state.outcome is None


async def test_already_paid_goes_to_disputes(make_state, ledger):
    state = make_state("CUST-002")
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Nova Retail"}), ("say", "ok"),
        ("tool", "transfer_to_disputes", {"invoice_number": "INV-1988"}), ("say", "I understand."),
        ("tool", "log_dispute", {"category": "already_paid", "details": "Paid by NEFT on the fifth"}),
        ("say", "Logged."),
        ("tool", "back_to_main", {}), ("say", "Anything else?"),
    ])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        await session.run(user_input="Nova Retail here")
        await session.run(user_input="We already paid that one.")
        assert isinstance(session.current_agent, DisputesAgent)
        r = await session.run(user_input="Yes, that's correct.")
        assert "UTR" in outputs(r)[0].output
        await session.run(user_input="No, that's all.")
        assert isinstance(session.current_agent, CollectionsAgent)
    row = ledger.conn.execute("SELECT invoice_number, category FROM disputes").fetchone()
    assert tuple(row) == ("INV-1988", "already_paid")
    assert state.outcome == "already_paid_claimed"


async def test_payment_link_and_plan_write_to_outbox_and_ledger(make_state, ledger):
    state = make_state("CUST-001")
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Shree Ganesh Traders"}), ("say", "ok"),
        ("tool", "transfer_to_payments", {}), ("say", "ok"),
        ("tool", "set_up_payment_plan", {"installments": 3,
                                         "first_payment_date": (TODAY + timedelta(days=3)).isoformat()}),
        ("say", "Done."),
        ("tool", "send_payment_link", {}), ("say", "Sent."),
    ])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        await session.run(user_input="Shree Ganesh Traders")
        await session.run(user_input="Can I pay in parts?")
        r = await session.run(user_input="Three parts, starting this Friday.")
        assert "three payments of forty-one thousand five hundred rupees and seventeen paise" in outputs(r)[0].output
        await session.run(user_input="Email me a link.")
    mail = ledger.conn.execute("SELECT to_email, subject FROM outbox").fetchone()
    assert tuple(mail) == ("accounts@shreeganesh.example", "Payment link for INV-2041")
    assert state.outcome == "payment_plan"


async def test_policy_tool_is_grounded(make_state):
    state = make_state("CUST-001")
    model = ScriptedLLM([("tool", "lookup_policy", {"question": "Is there a late fee?"}), ("say", "ok")])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        r = await session.run(user_input="Is there a late fee?")
    assert outputs(r)[0].output.startswith("[late-fees]")


async def test_promise_without_amount_means_full_balance_and_errors_give_the_number(make_state, ledger):
    """Models are bad at turning 'one lakh twenty-four thousand...' back into digits, so tools
    expose the exact number and a full-balance promise needs no amount at all."""
    state = make_state("CUST-001")
    on = (TODAY + timedelta(days=4)).isoformat()
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Shree Ganesh Traders"}), ("say", "ok"),
        ("tool", "transfer_to_payments", {}), ("say", "ok"),
        ("tool", "record_promise_to_pay", {"amount": 12450050, "payment_date": on}), ("say", "Let me fix that."),
        ("tool", "record_promise_to_pay", {"payment_date": on}), ("say", "Saved."),
    ])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        r0 = await session.run(user_input="Shree Ganesh Traders")
        assert "tool value 124500.50" in outputs(r0)[0].output
        await session.run(user_input="Full amount")
        r1 = await session.run(user_input="Yes")
        r2 = await session.run(user_input="Yes, the full amount")
    assert outputs(r1)[0].is_error and "tool value 124500.50" in outputs(r1)[0].output
    assert not outputs(r2)[0].is_error
    row = ledger.conn.execute("SELECT amount FROM promises").fetchone()
    assert Decimal(row["amount"]) == Decimal("124500.50")
