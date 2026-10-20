"""Place outbound collection calls by dispatching the agent into a new room.

    python -m duecall.dialer --customer CUST-001 --phone +919800000001
    python -m duecall.dialer --overdue            # everyone with an overdue invoice
    python -m duecall.dialer --overdue --ab v1,v2 # alternate prompt versions per call (A/B)

Needs LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET, and a running worker.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import uuid
from datetime import date

from dotenv import load_dotenv
from livekit import api

from .config import Settings
from .ledger import Ledger


def overdue_customers(ledger: Ledger, today: date) -> list[tuple[str, str]]:
    rows = ledger.conn.execute(
        "SELECT DISTINCT c.id, c.phone FROM customers c JOIN invoices i ON i.customer_id = c.id "
        "WHERE i.due_date < ? AND CAST(i.amount AS REAL) - CAST(i.paid AS REAL) > 0.005 "
        "AND c.phone IS NOT NULL ORDER BY c.id", (today.isoformat(),)).fetchall()
    return [(r["id"], r["phone"]) for r in rows]


def plan_calls(targets: list[tuple[str, str]], versions: list[str]) -> list[dict]:
    """Round-robin prompt versions over targets so each version sees a similar mix."""
    cycle = itertools.cycle(versions)
    return [{"customer_id": c, "phone": p, "prompt_version": next(cycle)} for c, p in targets]


async def dispatch(calls: list[dict], agent_name: str) -> None:
    async with api.LiveKitAPI() as lk:
        for meta in calls:
            room = f"call-{meta['customer_id'].lower()}-{uuid.uuid4().hex[:6]}"
            await lk.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
                agent_name=agent_name, room=room, metadata=json.dumps(meta)))
            print(f"dispatched {meta['customer_id']} ({meta['prompt_version']}) -> room {room}")


def main() -> None:
    load_dotenv()
    s = Settings()
    p = argparse.ArgumentParser()
    p.add_argument("--customer")
    p.add_argument("--phone")
    p.add_argument("--overdue", action="store_true")
    p.add_argument("--ab", default=s.prompt_version, help="comma-separated prompt versions")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    versions = a.ab.split(",")
    if a.overdue:
        ledger = Ledger(s.ledger_path)
        targets = overdue_customers(ledger, date.today())
    elif a.customer and a.phone:
        targets = [(a.customer, a.phone)]
    else:
        p.error("give --customer and --phone, or --overdue")
    calls = plan_calls(targets, versions)
    if a.dry_run:
        print(json.dumps(calls, indent=2))
        return
    asyncio.run(dispatch(calls, s.agent_name))


if __name__ == "__main__":
    main()
