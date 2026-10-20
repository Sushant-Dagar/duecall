"""Collections dashboard: call outcomes, money promised, disputes, outcome rate per
prompt version and voice latency, straight from the ledger with SQL.

    uvicorn duecall.dashboard:app --port 8000
"""

from __future__ import annotations

import html
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .ledger import Ledger


def create_app(ledger: Ledger | None = None) -> FastAPI:
    app = FastAPI(title="DueCall dashboard")
    led = ledger or Ledger(os.getenv("LEDGER_PATH", "data/ledger.db"))

    @app.get("/api/summary")
    def summary() -> dict:
        return led.summary()

    @app.get("/healthz")
    def health() -> dict:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        s = led.summary()

        def table(rows: list[dict], title: str) -> str:
            if not rows:
                return f"<h2>{title}</h2><p class=muted>No data yet.</p>"
            head = "".join(f"<th>{html.escape(k)}</th>" for k in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(v))}</td>" for v in r.values()) + "</tr>"
                           for r in rows)
            return f"<h2>{title}</h2><table><tr>{head}</tr>{body}</table>"

        p = s["promised_total"]
        return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>DueCall</title>
<style>body{{font:15px system-ui;max-width:960px;margin:24px auto;padding:0 16px;color:#1d2433}}
table{{border-collapse:collapse;width:100%;margin-bottom:8px}}td,th{{border-bottom:1px solid #e3e6ec;
padding:6px 8px;text-align:left}}th{{color:#5b6475;font-weight:600}}.kpi{{font-size:28px;font-weight:700}}
.muted{{color:#5b6475}}</style></head><body>
<h1>DueCall</h1><p class=muted>Voice collections: outcomes, commitments and latency.</p>
<div class=kpi>{p['n']} promises · {p['amount']:,.2f} committed</div>
{table(s['outcomes'], 'Outcomes')}{table(s['by_prompt'], 'By prompt version')}
{table(s['latency'], 'Voice latency (ms)')}{table(s['disputes'], 'Disputes')}
{table(s['recent_calls'], 'Recent calls')}</body></html>"""

    return app


def _fmt(v) -> str:
    return f"{v:,.1f}" if isinstance(v, float) else ("" if v is None else str(v))


app = create_app() if os.getenv("DUECALL_DASHBOARD_AUTOLOAD", "1") == "1" else None
