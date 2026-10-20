# DueCall: AI voice agent for accounts-receivable collections

DueCall phones a business customer about an overdue invoice, verifies who it is talking to, explains what is owed, and leaves the call with a recorded outcome: a promise to pay, a payment plan, a logged dispute, or a handover to a human. It runs on **LiveKit Agents** with streaming **speech-to-text → LLM → text-to-speech**, dials real phone numbers over **SIP telephony**, and writes every commitment to a SQL ledger that a finance team can act on.

```
 Phone (SIP trunk) ─┐
                    ▼
            LiveKit room ──► STT (Deepgram nova-3 | Google Chirp)
                    ▲              │  semantic turn detection + VAD
                    │              ▼
     TTS (Cartesia │ Google) ◄── LLM (Gemini | GPT | Claude)
       ▲  speech normaliser        │ function calls
       │                           ▼
       │      CollectionsAgent (supervisor: verify, triage)
       │        ├─► PaymentsAgent  (promise to pay, plan, payment link)
       │        └─► DisputesAgent  (log dispute, already-paid claims)
       │                           │
       └──────── tool results ◄── Ledger (SQLite/SQL) + policy RAG (BM25)
                                   │
                         dashboard · email outbox · latency metrics
```

## What makes it a voice agent and not a chatbot with a speaker

- **Every number is spoken right.** Amounts, dates and invoice references are converted to words in code (`speech.py`) before the LLM ever sees them: ₹1,24,500.50 becomes "one lakh twenty-four thousand five hundred rupees and fifty paise", `INV-2041` becomes "I N V, two zero four one". A sentence-buffered `tts_node` catches anything the model writes in digits anyway, without splitting an amount across TTS chunks.
- **Prompts written for the ear.** Prompt v2 asks one question per turn, keeps to one or two sentences, reads amounts and dates back before saving them, and recovers from interruptions. v1 is kept as the A/B baseline.
- **Latency is designed in.** Fixed opening line (no LLM round-trip when the customer answers), semantic end-of-turn detection, preemptive generation on interim transcripts, VAD loaded once per worker process. End-of-utterance delay, LLM time-to-first-token and TTS time-to-first-byte are logged per turn to the ledger and shown on the dashboard.
- **The call waits for a human.** Outbound SIP dialling blocks until the customer answers; unanswered calls are logged as `voicemail` instead of the agent talking to a ringing line.

## Guardrails enforced in code, not in the prompt

| Rule | Where |
|---|---|
| No invoice, amount or tool access before identity is verified | `_require_verified` in every account tool |
| Identity is a fuzzy match against **one** account, never a search across customers | `Ledger.match_company` |
| Two failed checks ends verification; nothing is disclosed | `verify_identity` |
| Promise must be > 0, ≤ balance, and within 14 days | `Ledger.record_promise` |
| Plans: 2–3 instalments, balance ≥ ₹50,000, first payment within 14 days | `Ledger.create_plan` |
| Invoices are always scoped to the called customer | `Ledger.invoice` |
| Specialists cannot write outside their job (payments can't log disputes) | separate agents and tool lists |

A rejected action comes back to the model as a `ToolError` with a sentence it can say to the customer ("I can only note a payment date within fourteen days…"), and nothing is written.

## Grounding

The model answers policy questions (late fees, payment methods, plans, disputes, credit hold) only through `lookup_policy`, a BM25 retriever over the policy knowledge base in `policy.py`. Off-topic questions return nothing, and the prompt tells the agent to offer a human instead of guessing. BM25 keeps retrieval inside a voice turn's latency budget; swap in a vector store when the knowledge base grows.

## Evaluation and prompt A/B testing

`evals/run_eval.py` runs simulated calls: an LLM plays the customer (cooperative, needs a plan, already paid, quantity dispute, wrong company, policy questions, wants a human) against each prompt version, through the same `AgentSession` and tools as a real call. Each transcript is scored on:

- **outcome accuracy**: did the call end in an acceptable outcome for the scenario
- **hallucinated amounts**: every amount the agent *said* (parsed back from spoken words) must be one a tool disclosed on that call
- **leaks before verification**: amounts spoken before identity was confirmed
- **TTS violations**: digits, symbols or raw references in agent speech
- **words per turn**: brevity on the phone
- **LLM judge** (optional `--judge`): a self-reflection pass scoring the transcript against a rubric

```bash
python -m evals.run_eval --versions v1,v2 --repeats 3 --judge
```

Results are written to `evals/results/` with each prompt version's content hash, so a score always maps back to the exact prompt text. The dialer can also alternate prompt versions across live calls (`--ab v1,v2`), and the dashboard reports commitment rate per version.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # LiveKit + model keys
python -m livekit.agents download-files

pytest -q                         # offline: no keys needed
python -m duecall.main console    # talk to the agent through your mic
python -m duecall.main dev        # connect to LiveKit Cloud (test from the Agents Playground)

# real phone calls (needs a LiveKit outbound SIP trunk)
python -m duecall.dialer --customer CUST-001 --phone +91XXXXXXXXXX
python -m duecall.dialer --overdue --ab v1,v2 --dry-run

uvicorn duecall.dashboard:app --port 8000   # outcomes, promises, disputes, latency
```

Swap providers with environment variables, no code changes: `LLM_PROVIDER=google|openai|anthropic`, `STT_PROVIDER=deepgram|google`, `TTS_PROVIDER=cartesia|google`.

## Deploy on Google Cloud Run

```bash
PROJECT=my-gcp-project REGION=asia-south1 LIVEKIT_URL=wss://... ./deploy/cloudrun.sh
```

The worker keeps a websocket open to LiveKit and must not be CPU-throttled, so the script deploys with `--no-cpu-throttling --min-instances 1`, and keys come from Secret Manager.

## Tests

41 offline tests (`pytest -q`, about a second). The conversation tests run the real LiveKit `AgentSession`, tool execution and agent handoffs with a scripted stand-in LLM (`scripted_llm.py`), so CI checks the call flow and every guardrail without API keys. Model quality is measured separately by the eval harness.

## Layout

```
duecall/
  agents.py        supervisor + payments + disputes agents, tools, TTS-safe output
  speech.py        amounts / dates / references → spoken text; streaming normaliser
  ledger.py        SQL ledger: customers, invoices, promises, plans, disputes, outbox, metrics
  policy.py        hard limits + policy knowledge base + BM25 retriever
  prompts.py       versioned prompts (v1 baseline, v2 voice-optimised) with content hashes
  main.py          LiveKit worker: SIP dial-out, STT/LLM/TTS session, latency metrics
  dialer.py        dispatch outbound calls, A/B prompt assignment
  dashboard.py     FastAPI dashboard over the ledger
  evaluation.py    scoring, simulated customer, LLM judge
  scripted_llm.py  deterministic LLM for offline tests
evals/run_eval.py  prompt A/B eval CLI
deploy/            Cloud Run deploy script
```

Demo data: three accounts (`CUST-001` two invoices, one overdue; `CUST-002` part-paid and long overdue; `CUST-003` in SGD, below the plan threshold). Company and customer names are fictional.
