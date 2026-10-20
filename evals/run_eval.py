"""A/B evaluate prompt versions on simulated calls.

    python -m evals.run_eval --versions v1,v2 --repeats 3 --judge

Uses the LLM set by LLM_PROVIDER / LLM_MODEL for the agent, and the same provider
for the simulated customer and the judge (override with --sim-provider).
Writes evals/results/<timestamp>.md and .json.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

from duecall import prompts
from duecall.agents import CallState
from duecall.config import Settings, make_llm
from duecall.evaluation import Scenario, dump, judge, llm_customer, run_call, score, summarize, to_markdown
from duecall.ledger import Ledger

SCENARIOS = [
    Scenario("cooperative", "CUST-001",
             "Rohit Mehta, accounts at Shree Ganesh Traders. Friendly. Can pay the full overdue invoice "
             "next Friday once a client pays you.", frozenset({"promise_to_pay"})),
    Scenario("needs-plan", "CUST-001",
             "Rohit Mehta at Shree Ganesh Traders. Cash is tight this month. Cannot pay the overdue invoice in one "
             "go, but could pay in three parts starting next week.", frozenset({"payment_plan"})),
    Scenario("already-paid", "CUST-002",
             "Anjali Rao, accounts at Nova Retail. Annoyed: you are sure invoice one nine eight eight was paid in "
             "full by bank transfer two weeks ago.", frozenset({"already_paid_claimed", "dispute_logged"})),
    Scenario("quantity-dispute", "CUST-003",
             "Daniel Tan at Bluepeak Logistics. Only received half the shipment for the overdue invoice, so you "
             "won't pay until it is corrected.", frozenset({"dispute_logged"})),
    Scenario("wrong-company", "CUST-002",
             "You work at Sunrise Hardware, not Nova Retail. You picked up a forwarded call by mistake and keep "
             "saying Sunrise Hardware when asked.", frozenset({"verification_failed", "wrong_person"})),
    Scenario("policy-questions", "CUST-001",
             "Rohit Mehta at Shree Ganesh Traders. Before committing, asks whether there is a late fee and whether "
             "you can pay by UPI. Then promises to pay in full in ten days.", frozenset({"promise_to_pay"})),
    Scenario("wants-human", "CUST-002",
             "Anjali Rao at Nova Retail. Refuses to discuss payment with an automated system and insists on speaking "
             "to a person in the finance team.", frozenset({"escalated", "callback_requested"})),
]


async def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--versions", default="v1,v2")
    p.add_argument("--scenarios", default="all")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--sim-provider", default=None)
    a = p.parse_args()

    s = Settings()
    agent_llm = make_llm(s)
    sim_llm = make_llm(replace(s, llm_provider=a.sim_provider, llm_model=None) if a.sim_provider else s)
    chosen = SCENARIOS if a.scenarios == "all" else [x for x in SCENARIOS if x.id in a.scenarios.split(",")]
    today = date.today()

    def make_state(customer_id: str, version: str) -> CallState:
        ledger = Ledger(":memory:")
        ledger.seed_demo(today)
        call_id = ledger.start_call(customer_id, version, "eval")
        return CallState(ledger=ledger, customer_id=customer_id, call_id=call_id, today=today, prompt_version=version)

    scores = []
    for version in a.versions.split(","):
        for sc in chosen:
            for _ in range(a.repeats):
                contact = Ledger(":memory:")
                contact.seed_demo(today)
                opening = prompts.opening_line(contact.customer(sc.customer_id).contact_name)
                t = await run_call(sc, version, agent_llm, llm_customer(sim_llm, sc.persona), make_state,
                                   opening=opening)
                sc_score = score(t, sc)
                if a.judge:
                    sc_score.judge, _ = await judge(sim_llm, t)
                scores.append(sc_score)
                print(f"{version:>3} {sc.id:<17} outcome={t.outcome} ok={sc_score.outcome_ok} "
                      f"halluc={len(sc_score.hallucinations)} tts={len(sc_score.tts_violations)}")

    summary = summarize(scores)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    header = (f"# Eval {stamp}\n\nmodel: {s.llm_provider}/{s.llm_model or 'default'}; "
              + ", ".join(f"{v}@{prompts.fingerprint(v)}" for v in a.versions.split(",")) + "\n\n")
    (out / f"{stamp}.md").write_text(header + to_markdown(summary, scores) + "\n")
    (out / f"{stamp}.json").write_text(dump(scores))
    print("\n" + to_markdown(summary, scores))
    print(f"\nsaved evals/results/{stamp}.md")
    os._exit(0)  # plugin HTTP sessions can keep the loop alive


if __name__ == "__main__":
    asyncio.run(main())
