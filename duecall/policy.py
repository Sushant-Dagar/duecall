"""Collections policy: the hard limits (enforced in code) and the knowledge base
(retrieved for grounding answers to customer questions).

The two are kept apart on purpose. Limits like "plans max out at three payments"
are checked by the ledger whatever the model says. The knowledge base is what the
model reads from when a customer asks "what happens if I pay late?", so its answer
is grounded in a passage instead of a guess.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal

PROMISE_MAX_DAYS = 14
PLAN_MAX_INSTALLMENTS = 3
PLAN_MIN_BALANCE = Decimal("50000")
NAME_MATCH_THRESHOLD = 0.72
MAX_VERIFY_ATTEMPTS = 2

OUTCOMES = {
    "promise_to_pay", "payment_plan", "dispute_logged", "escalated", "already_paid_claimed",
    "callback_requested", "wrong_person", "verification_failed", "no_commitment", "voicemail",
}
DISPUTE_CATEGORIES = {"pricing", "quantity", "quality", "not_received", "already_paid", "other"}


KB = {
    "late-fees": "Invoices paid after the due date attract a late fee of one and a half percent per month on "
                 "the unpaid balance. The fee is waived if the customer pays in full within seven days of the call.",
    "payment-methods": "Customers can pay by bank transfer (NEFT, RTGS or IMPS), UPI, or the secure payment link "
                       "we email. We never ask for card details, OTPs or passwords over the phone.",
    "payment-plans": "Balances of fifty thousand rupees or more can be split into two or three instalments. "
                     "The first instalment must be paid within fourteen days. Plans need no extra paperwork.",
    "promise-window": "A promise to pay can be recorded for any date within the next fourteen days. "
                      "Later dates need a payment plan or approval from the finance team.",
    "disputes": "If a customer disputes an invoice for pricing, quantity, quality or non-delivery, collection on "
                "that invoice pauses while the finance team reviews it. The team replies by email within three "
                "working days.",
    "already-paid": "If a customer says they already paid, we log it as a dispute of type already paid and ask "
                    "them to email the payment reference or UTR number. Reconciliation takes up to two working days.",
    "credit-hold": "Accounts with invoices more than sixty days overdue may be placed on credit hold, which pauses "
                   "new orders until the balance is cleared or a plan is agreed.",
    "contact-hours": "The finance team is available Monday to Friday, ten in the morning to six in the evening, "
                     "India time. A human callback can be requested at any time during the call.",
    "statements": "Customers can ask for an account statement listing all open invoices. It is sent by email "
                  "to the address on file, never to a new address given on the call.",
}


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    score: float


_TOKEN = re.compile(r"[a-z]+")
_STOP = set("a an the is are be to of and or in on for if we our you your it can i my me do does what how "
            "with that this by at any from".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class PolicyIndex:
    """BM25 over the policy passages. Small, offline and fast enough to run inside a
    voice turn without adding an embedding round-trip; swap in a vector store when
    the knowledge base outgrows a few hundred passages."""

    def __init__(self, docs: dict[str, str] = KB, k1: float = 1.4, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = {i: Counter(_tokens(t)) for i, t in docs.items()}
        self.len = {i: sum(c.values()) for i, c in self.tf.items()}
        self.avg = sum(self.len.values()) / len(self.len)
        df = Counter(t for c in self.tf.values() for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 2, min_score: float = 0.5) -> list[Passage]:
        q = _tokens(query)
        scored = []
        for i, tf in self.tf.items():
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s >= min_score:
                scored.append(Passage(i, self.docs[i], round(s, 3)))
        return sorted(scored, key=lambda p: -p.score)[:k]
