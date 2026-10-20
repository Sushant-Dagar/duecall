"""Accounts-receivable ledger the agent reads from and writes to (SQLite).

The ledger is the agent's only source of truth for money. The LLM never states an
amount or due date it did not get from here, and nothing it promises on a call is
written unless these checks pass. Validation errors are raised as LedgerError with
a sentence the agent can say back to the customer.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path

from . import policy

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY, company TEXT NOT NULL, contact_name TEXT NOT NULL,
    phone TEXT, email TEXT, currency TEXT NOT NULL DEFAULT 'INR');
CREATE TABLE IF NOT EXISTS invoices (
    number TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
    issue_date TEXT NOT NULL, due_date TEXT NOT NULL,
    amount TEXT NOT NULL, paid TEXT NOT NULL DEFAULT '0');
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY, customer_id TEXT, started_at TEXT NOT NULL, ended_at TEXT,
    outcome TEXT, prompt_version TEXT, channel TEXT);
CREATE TABLE IF NOT EXISTS promises (
    id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, invoice_number TEXT NOT NULL,
    amount TEXT NOT NULL, promised_date TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS payment_plans (
    id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, invoice_number TEXT NOT NULL,
    installments INTEGER NOT NULL, installment_amount TEXT NOT NULL, first_date TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS disputes (
    id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, invoice_number TEXT NOT NULL,
    category TEXT NOT NULL, details TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, to_email TEXT NOT NULL,
    subject TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS turn_metrics (
    id INTEGER PRIMARY KEY, call_id TEXT NOT NULL, metric TEXT NOT NULL, value_ms REAL NOT NULL);
"""


class LedgerError(ValueError):
    """A rule was broken. The message is safe to read to the customer."""


@dataclass(frozen=True)
class Invoice:
    number: str
    customer_id: str
    issue_date: date
    due_date: date
    amount: Decimal
    paid: Decimal

    @property
    def balance(self) -> Decimal:
        return self.amount - self.paid

    def days_overdue(self, today: date) -> int:
        return max(0, (today - self.due_date).days)


@dataclass(frozen=True)
class Customer:
    id: str
    company: str
    contact_name: str
    phone: str | None
    email: str | None
    currency: str


def _d(s: str) -> date:
    return date.fromisoformat(s)


class Ledger:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---- reads --------------------------------------------------------------
    def customer(self, customer_id: str) -> Customer | None:
        row = self.conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        return Customer(**dict(row)) if row else None

    def match_company(self, customer_id: str, spoken_name: str) -> bool:
        """Identity check: does the name the person said match the account we called?

        Speech-to-text mangles company names ("Shree Ganesh" -> "Shri Ganesh"), so
        this is a fuzzy match, but against ONE known account, never a search across
        all customers, so a caller cannot fish for someone else's balance.
        """
        cust = self.customer(customer_id)
        if not cust:
            return False

        def norm(s: str) -> str:
            s = s.lower()
            for junk in (" private limited", " pvt ltd", " pvt. ltd.", " limited", " ltd", " llp", "."):
                s = s.replace(junk, "")
            return " ".join(s.split())

        return SequenceMatcher(None, norm(cust.company), norm(spoken_name)).ratio() >= policy.NAME_MATCH_THRESHOLD

    def invoice(self, number: str, customer_id: str) -> Invoice:
        row = self.conn.execute(
            "SELECT * FROM invoices WHERE upper(number) = upper(?) AND customer_id = ?",
            (number.strip(), customer_id)).fetchone()
        if not row:
            raise LedgerError(f"I can't find invoice {number} on this account.")
        return self._inv(row)

    def open_invoices(self, customer_id: str) -> list[Invoice]:
        rows = self.conn.execute(
            "SELECT * FROM invoices WHERE customer_id = ? "
            "AND CAST(amount AS REAL) - CAST(paid AS REAL) > 0.005 ORDER BY due_date",
            (customer_id,)).fetchall()
        return [self._inv(r) for r in rows]

    @staticmethod
    def _inv(row: sqlite3.Row) -> Invoice:
        return Invoice(row["number"], row["customer_id"], _d(row["issue_date"]), _d(row["due_date"]),
                       Decimal(row["amount"]), Decimal(row["paid"]))

    # ---- writes (all validated) ---------------------------------------------
    def start_call(self, customer_id: str | None, prompt_version: str, channel: str) -> str:
        call_id = "call_" + uuid.uuid4().hex[:10]
        self.conn.execute(
            "INSERT INTO calls (id, customer_id, started_at, prompt_version, channel) VALUES (?,?,?,?,?)",
            (call_id, customer_id, datetime.now().isoformat(timespec="seconds"), prompt_version, channel))
        self.conn.commit()
        return call_id

    def end_call(self, call_id: str, outcome: str) -> None:
        if outcome not in policy.OUTCOMES:
            raise LedgerError(f"Unknown outcome {outcome!r}.")
        self.conn.execute("UPDATE calls SET ended_at = ?, outcome = COALESCE(outcome, ?) WHERE id = ?",
                          (datetime.now().isoformat(timespec="seconds"), outcome, call_id))
        self.conn.commit()

    def record_promise(self, call_id: str, inv: Invoice, amount: Decimal, on: date, today: date) -> None:
        if amount <= 0:
            raise LedgerError("The amount has to be more than zero.")
        if amount > inv.balance:
            raise LedgerError("That is more than the balance due on this invoice.")
        if on < today:
            raise LedgerError("That date has already passed.")
        if on > today + timedelta(days=policy.PROMISE_MAX_DAYS):
            raise LedgerError(
                f"I can only note a payment date within {policy.PROMISE_MAX_DAYS} days. "
                "For longer, we can look at a payment plan.")
        self.conn.execute(
            "INSERT INTO promises (call_id, invoice_number, amount, promised_date, created_at) VALUES (?,?,?,?,?)",
            (call_id, inv.number, str(amount), on.isoformat(), datetime.now().isoformat(timespec="seconds")))
        self.conn.commit()

    def create_plan(self, call_id: str, inv: Invoice, installments: int, first: date, today: date) -> Decimal:
        if inv.balance < policy.PLAN_MIN_BALANCE:
            raise LedgerError("This balance is too small for a payment plan; it needs to be paid in one go.")
        if not 2 <= installments <= policy.PLAN_MAX_INSTALLMENTS:
            raise LedgerError(f"A plan can be split into two to {policy.PLAN_MAX_INSTALLMENTS} payments.")
        if not today <= first <= today + timedelta(days=policy.PROMISE_MAX_DAYS):
            raise LedgerError(f"The first payment needs to be within {policy.PROMISE_MAX_DAYS} days.")
        each = (inv.balance / installments).quantize(Decimal("0.01"))
        self.conn.execute(
            "INSERT INTO payment_plans (call_id, invoice_number, installments, installment_amount, first_date) "
            "VALUES (?,?,?,?,?)", (call_id, inv.number, installments, str(each), first.isoformat()))
        self.conn.commit()
        return each

    def log_dispute(self, call_id: str, inv: Invoice, category: str, details: str) -> None:
        if category not in policy.DISPUTE_CATEGORIES:
            raise LedgerError(f"Unknown dispute category {category!r}.")
        self.conn.execute("INSERT INTO disputes (call_id, invoice_number, category, details) VALUES (?,?,?,?)",
                          (call_id, inv.number, category, details.strip()[:1000]))
        self.conn.commit()

    def escalate(self, call_id: str, reason: str) -> None:
        self.conn.execute("INSERT INTO escalations (call_id, reason) VALUES (?,?)", (call_id, reason[:500]))
        self.conn.commit()

    def queue_email(self, call_id: str, to: str, subject: str, body: str) -> None:
        self.conn.execute(
            "INSERT INTO outbox (call_id, to_email, subject, body, created_at) VALUES (?,?,?,?,?)",
            (call_id, to, subject, body, datetime.now().isoformat(timespec="seconds")))
        self.conn.commit()

    def record_metric(self, call_id: str, metric: str, value_ms: float) -> None:
        self.conn.execute("INSERT INTO turn_metrics (call_id, metric, value_ms) VALUES (?,?,?)",
                          (call_id, metric, value_ms))
        self.conn.commit()

    # ---- seed ----------------------------------------------------------------
    def seed_demo(self, today: date) -> None:
        """Three demo accounts, each set up to exercise a different path."""
        customers = [
            ("CUST-001", "Shree Ganesh Traders", "Rohit Mehta", "+919800000001", "accounts@shreeganesh.example", "INR"),
            ("CUST-002", "Nova Retail Pvt Ltd", "Anjali Rao", "+919800000002", "ap@novaretail.example", "INR"),
            ("CUST-003", "Bluepeak Logistics", "Daniel Tan", "+6580000003", "finance@bluepeak.example", "SGD"),
        ]
        invoices = [
            ("INV-2041", "CUST-001", -55, -25, "124500.50", "0"),
            ("INV-2077", "CUST-001", -20, 10, "38000", "0"),      # not yet due
            ("INV-1988", "CUST-002", -75, -45, "86250", "20000"),  # part paid, long overdue
            ("INV-3102", "CUST-003", -40, -10, "4800", "0"),
        ]
        self.conn.executemany("INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?,?)", customers)
        self.conn.executemany(
            "INSERT OR REPLACE INTO invoices VALUES (?,?,?,?,?,?)",
            [(n, c, (today + timedelta(days=i)).isoformat(), (today + timedelta(days=d)).isoformat(), a, p)
             for n, c, i, d, a, p in invoices])
        self.conn.commit()

    # ---- reporting -------------------------------------------------------------
    def summary(self) -> dict:
        q = lambda sql: [dict(r) for r in self.conn.execute(sql).fetchall()]  # noqa: E731
        return {
            "outcomes": q("SELECT COALESCE(outcome,'in_progress') AS outcome, COUNT(*) AS calls "
                          "FROM calls GROUP BY 1 ORDER BY 2 DESC"),
            "promised_total": q("SELECT COUNT(*) AS n, COALESCE(SUM(CAST(amount AS REAL)),0) AS amount FROM promises")[0],
            "disputes": q("SELECT category, COUNT(*) AS n FROM disputes GROUP BY 1 ORDER BY 2 DESC"),
            "by_prompt": q("SELECT prompt_version, COUNT(*) AS calls, "
                           "SUM(outcome IN ('promise_to_pay','payment_plan')) AS committed "
                           "FROM calls GROUP BY 1"),
            "latency": q("SELECT metric, COUNT(*) AS n, AVG(value_ms) AS avg_ms, MAX(value_ms) AS max_ms "
                         "FROM turn_metrics GROUP BY 1 ORDER BY 1"),
            "recent_calls": q("SELECT c.id, cu.company, c.outcome, c.prompt_version, c.started_at FROM calls c "
                              "LEFT JOIN customers cu ON cu.id = c.customer_id ORDER BY c.started_at DESC LIMIT 20"),
        }
