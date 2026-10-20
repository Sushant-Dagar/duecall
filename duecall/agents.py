"""The voice agents: a supervisor that verifies and triages, and two specialists it
hands off to (payments, disputes).

Splitting the call this way keeps each agent's prompt and tool list small, which
helps both accuracy and latency: the payments agent cannot log a dispute and the
dispute agent cannot record a promise, so a confused turn cannot write to the
wrong place. All three share the same CallState, so nothing is lost on handoff.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from livekit.agents import Agent, ModelSettings, RunContext, ToolError, function_tool
from pydantic import Field

from . import policy, prompts
from .ledger import Invoice, Ledger, LedgerError
from .speech import SentenceBuffer, number_to_words, speak_amount, speak_date, speak_reference


@dataclass
class CallState:
    ledger: Ledger
    customer_id: str
    call_id: str
    today: date
    prompt_version: str = prompts.DEFAULT_VERSION
    verified: bool = False
    verify_attempts: int = 0
    focus_invoice: str | None = None
    outcome: str | None = None
    # Every amount and date a tool disclosed on this call. The eval harness checks
    # the agent's speech against this set to catch hallucinated figures.
    disclosed_amounts: set[Decimal] = field(default_factory=set)
    disclosed_dates: set[date] = field(default_factory=set)

    @property
    def currency(self) -> str:
        cust = self.ledger.customer(self.customer_id)
        return cust.currency if cust else "INR"

    def amount(self, value: Decimal) -> str:
        self.disclosed_amounts.add(value.quantize(Decimal("0.01")))
        return speak_amount(value, self.currency)

    def day(self, d: date) -> str:
        self.disclosed_dates.add(d)
        return speak_date(d, today=self.today)

    def describe(self, inv: Invoice) -> str:
        overdue = inv.days_overdue(self.today)
        if overdue:
            status = f"{_small(overdue)} day{'s' if overdue != 1 else ''} overdue"
        else:
            status = f"due on {self.day(inv.due_date)}"
        # The spoken form is for the customer; the "tool value" is the exact number the
        # model passes back to tools, so it never has to convert words to digits itself.
        return (f"invoice {speak_reference(inv.number)} (tool value {inv.number}), balance "
                f"{self.amount(inv.balance)} (tool value {inv.balance}), {status}")


def _small(n: int) -> str:
    return number_to_words(n)


Ctx = RunContext[CallState]


def _require_verified(state: CallState) -> None:
    if not state.verified:
        raise ToolError("Identity is not verified yet. Ask which company they are with and call verify_identity.")


def _parse_date(value: str, state: CallState) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ToolError(f"Could not read the date {value!r}. Resolve it against today ({state.today}) "
                        "and pass it as YYYY-MM-DD.") from None


def _parse_amount(value: float | str) -> Decimal:
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except InvalidOperation:
        raise ToolError(f"Could not read the amount {value!r}.") from None


class _VoiceAgent(Agent):
    """Shared behaviour: TTS-safe output and the tools every agent needs."""

    role = "collections"

    def __init__(self, state: CallState, *, chat_ctx=None) -> None:
        super().__init__(
            instructions=prompts.get(state.prompt_version, self.role, today=state.today.isoformat()),
            chat_ctx=chat_ctx,
        )
        self.state = state

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        # Safety net: even if the model writes "Rs 4,800" or "INV-3102", the
        # customer hears it spoken correctly. Buffered per sentence.
        buf = SentenceBuffer(currency=self.state.currency, today=self.state.today)

        async def normalized() -> AsyncIterable[str]:
            async for chunk in text:
                for sentence in buf.push(chunk):
                    yield sentence
            for sentence in buf.flush():
                yield sentence

        return Agent.default.tts_node(self, normalized(), model_settings)

    @function_tool
    async def lookup_policy(self, context: Ctx, question: str) -> str:
        """Look up the collections policy (late fees, payment methods, plans, disputes, credit hold,
        contact hours, statements). Answer the customer only from what this returns."""
        hits = policy.PolicyIndex().search(question)
        if not hits:
            return "No policy covers this. Tell the customer you will have the finance team confirm, and offer request_human."
        return "\n".join(f"[{p.id}] {p.text}" for p in hits)

    @function_tool
    async def request_human(self, context: Ctx, reason: str) -> str:
        """Escalate to a human from the finance team, when the customer asks for one or the issue is beyond policy."""
        st = context.userdata
        st.ledger.escalate(st.call_id, reason)
        st.outcome = st.outcome or "escalated"
        return ("Escalation logged. Tell the customer a member of the finance team will call back within one "
                "working day, then call end_call.")

    @function_tool
    async def end_call(
        self,
        context: Ctx,
        outcome: Literal["promise_to_pay", "payment_plan", "dispute_logged", "escalated", "already_paid_claimed",
                         "callback_requested", "wrong_person", "verification_failed", "no_commitment", "voicemail"],
    ) -> str:
        """Close the call with its outcome, after saying a short goodbye."""
        st = context.userdata
        final = st.outcome or outcome
        st.ledger.end_call(st.call_id, final)
        st.outcome = final
        return f"Call closed as {final}. Say a brief thank-you and goodbye; do not ask further questions."


class CollectionsAgent(_VoiceAgent):
    """Supervisor: verifies the person, explains what is overdue, routes to a specialist."""

    role = "collections"

    @function_tool
    async def verify_identity(self, context: Ctx, company_name: str) -> str:
        """Verify the person by the company name they say they are calling from."""
        st = context.userdata
        if st.verified:
            return "Already verified."
        if st.ledger.match_company(st.customer_id, company_name):
            st.verified = True
            invoices = st.ledger.open_invoices(st.customer_id)
            if not invoices:
                st.outcome = "no_commitment"
                return "Verified. There are no open invoices; thank them and call end_call."
            lines = "; ".join(st.describe(i) for i in invoices)
            st.focus_invoice = invoices[0].number
            return f"Verified. Open invoices: {lines}. Tell them about these and ask when they can pay."
        st.verify_attempts += 1
        if st.verify_attempts >= policy.MAX_VERIFY_ATTEMPTS:
            st.outcome = "verification_failed"
            return ("Verification failed twice. Do not share any account details. Say you will send the details "
                    "by email to the address on file, then call end_call with verification_failed.")
        return "That company name does not match this account. Politely ask them to repeat the company name."

    @function_tool
    async def get_open_invoices(self, context: Ctx) -> str:
        """List the open invoices on this account with balances and how overdue they are."""
        st = context.userdata
        _require_verified(st)
        invoices = st.ledger.open_invoices(st.customer_id)
        return "; ".join(st.describe(i) for i in invoices) or "No open invoices."

    def _focus(self, st: CallState, invoice_number: str | None) -> Invoice:
        _require_verified(st)
        number = invoice_number or st.focus_invoice
        if not number:
            raise ToolError("Ask which invoice they mean.")
        try:
            inv = st.ledger.invoice(number, st.customer_id)
        except LedgerError as e:
            raise ToolError(str(e)) from None
        st.focus_invoice = inv.number
        return inv

    @function_tool
    async def transfer_to_payments(self, context: Ctx, invoice_number: str | None = None):
        """Hand off to the payments specialist to agree a payment date, a payment plan, or send a payment link.
        invoice_number like INV-2041; omit to use the invoice already discussed."""
        st = context.userdata
        self._focus(st, invoice_number)
        return PaymentsAgent(st, chat_ctx=self.chat_ctx), "Transferring to payments."

    @function_tool
    async def transfer_to_disputes(self, context: Ctx, invoice_number: str | None = None):
        """Hand off to the disputes specialist when the customer disagrees with an invoice or says it is paid."""
        st = context.userdata
        self._focus(st, invoice_number)
        return DisputesAgent(st, chat_ctx=self.chat_ctx), "Transferring to disputes."

    @function_tool
    async def send_statement(self, context: Ctx) -> str:
        """Email an account statement of all open invoices to the address on file."""
        st = context.userdata
        _require_verified(st)
        cust = st.ledger.customer(st.customer_id)
        rows = "\n".join(f"{i.number}: {i.balance} {cust.currency} due {i.due_date}"
                         for i in st.ledger.open_invoices(st.customer_id))
        st.ledger.queue_email(st.call_id, cust.email, f"Account statement - {cust.company}", rows)
        return "Statement emailed to the address on file. Tell them it is on its way."


class _Specialist(_VoiceAgent):
    def _inv(self, st: CallState) -> Invoice:
        _require_verified(st)
        if not st.focus_invoice:
            raise ToolError("No invoice in focus; call back_to_main.")
        return st.ledger.invoice(st.focus_invoice, st.customer_id)

    @function_tool
    async def back_to_main(self, context: Ctx):
        """Return to the main agent for anything outside this specialist's job."""
        return CollectionsAgent(context.userdata, chat_ctx=self.chat_ctx), "Back to the main agent."


class PaymentsAgent(_Specialist):
    role = "payments"

    @function_tool
    async def record_promise_to_pay(
        self, context: Ctx,
        payment_date: Annotated[str, Field(description="Date as YYYY-MM-DD")],
        amount: Annotated[float | None, Field(
            description="Amount as a number, e.g. the balance's tool value. Omit to promise the full balance.")] = None,
    ) -> str:
        """Save the customer's confirmed promise to pay. Only call after reading it back and getting a yes."""
        st = context.userdata
        inv = self._inv(st)
        amt = inv.balance if amount is None else _parse_amount(amount)
        on = _parse_date(payment_date, st)
        try:
            st.ledger.record_promise(st.call_id, inv, amt, on, st.today)
        except LedgerError as e:
            raise ToolError(f"{e} The balance is {speak_amount(inv.balance, st.currency)} "
                            f"(tool value {inv.balance}); omit amount for the full balance.") from None
        st.outcome = "promise_to_pay"
        left = inv.balance - amt
        tail = f" A balance of {st.amount(left)} will remain." if left > 0 else ""
        return (f"Saved: {st.amount(amt)} on {st.day(on)} for invoice {speak_reference(inv.number)}.{tail} "
                "Confirm this to the customer and offer a payment link.")

    @function_tool
    async def set_up_payment_plan(
        self, context: Ctx, installments: int,
        first_payment_date: Annotated[str, Field(description="Date as YYYY-MM-DD")],
    ) -> str:
        """Split the invoice balance into two or three equal instalments, monthly from the first date."""
        st = context.userdata
        inv = self._inv(st)
        first = _parse_date(first_payment_date, st)
        try:
            each = st.ledger.create_plan(st.call_id, inv, installments, first, st.today)
        except LedgerError as e:
            raise ToolError(str(e)) from None
        st.outcome = "payment_plan"
        return (f"Plan saved: {_small(installments)} payments of {st.amount(each)}, the first on {st.day(first)}, "
                "then monthly. Confirm it to the customer.")

    @function_tool
    async def send_payment_link(self, context: Ctx) -> str:
        """Email a secure payment link for the invoice in focus to the address on file."""
        st = context.userdata
        inv = self._inv(st)
        cust = st.ledger.customer(st.customer_id)
        st.ledger.queue_email(
            st.call_id, cust.email, f"Payment link for {inv.number}",
            f"Pay {inv.balance} {cust.currency} for {inv.number}: https://pay.example/{inv.number}")
        return "Payment link emailed to the address on file. Tell them to check their inbox."


class DisputesAgent(_Specialist):
    role = "disputes"

    @function_tool
    async def log_dispute(
        self, context: Ctx,
        category: Literal["pricing", "quantity", "quality", "not_received", "already_paid", "other"],
        details: Annotated[str, Field(description="One-sentence summary in the customer's words")],
    ) -> str:
        """Log a confirmed dispute on the invoice in focus. Collection pauses while finance reviews it."""
        st = context.userdata
        inv = self._inv(st)
        try:
            st.ledger.log_dispute(st.call_id, inv, category, details)
        except LedgerError as e:
            raise ToolError(str(e)) from None
        st.outcome = "already_paid_claimed" if category == "already_paid" else "dispute_logged"
        extra = " Ask them to email the payment reference or UTR number." if category == "already_paid" else ""
        return ("Dispute logged. Collection on this invoice is paused and finance will reply by email within "
                f"three working days.{extra}")
